import os
from feat import Detector
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder as label_encoder

# Initialize the feat Detector
detector = Detector()

# Process the captured image to extract AUs
def extract_aus(image_path):
    # Detect features
    results = detector.detect_image(image_path)

    # Expected features (must match the training features)
    expected_features = ["valence", "arousal"] + [
        "AU01", "AU02", "AU04", "AU05", "AU06", "AU07", "AU09",
        "AU10", "AU11", "AU12", "AU14", "AU15", "AU17", "AU20",
        "AU23", "AU24", "AU25", "AU26", "AU28", "AU43"
    ]

    # Initialize all features to 0
    feature_dict = {feature: 0.0 for feature in expected_features}

    # Update detected AUs
    detected_columns = results.filter(like="AU").columns
    for col in detected_columns:
        if col in feature_dict:
            feature_dict[col] = results[col].iloc[0]  # Use the first row for single image

    # Handle valence/arousal if available
    if "valence" in results.columns:
        feature_dict["valence"] = results["valence"].iloc[0]
    if "arousal" in results.columns:
        feature_dict["arousal"] = results["arousal"].iloc[0]

    # Convert to NumPy array with correct order
    features = np.array([feature_dict[feature] for feature in expected_features]).reshape(1, -1)
    return features






# Main process
if __name__ == "__main__":
    
    # Step 1 removed: take picture
    image_path = "captured_image.jpg"  # Path to the captured image

    # Step 2: Extract AUs from the captured image
    if os.path.exists(image_path):
        print("Extracting Action Units...")
        au_features = extract_aus(image_path)
        print("Extracted AUs:", au_features)

        # Example: Use the loaded model to predict emotion
        model_path = "random_forest_model.joblib"  # Ensure your model is saved at this path
        if os.path.exists(model_path):
            # Load the saved model
            loaded_model = joblib.load(model_path)
            print("Model loaded successfully!")

            # Load the saved scaler
            # Normalize the extracted features
            scaler_path = "scaler.pkl"
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    fitted_scaler = joblib.load(f)
                au_features_normalized = fitted_scaler.transform(au_features)
            else:
                raise FileNotFoundError("Scaler file not found. Train and save the scaler first.")



            # Normalize the extracted features
            au_features_normalized = fitted_scaler.transform(au_features)

            # Predict emotion
            # Load the LabelEncoder instance
            label_encoder_path = "label_encoder.pkl"
            if os.path.exists(label_encoder_path):
                with open(label_encoder_path, 'rb') as f:
                    loaded_label_encoder = joblib.load(f)
            else:
                raise FileNotFoundError("LabelEncoder file not found. Train and save the encoder first.")

            # Predict emotion probabilities
            predicted_probabilities = loaded_model.predict_proba(au_features_normalized)

            # Get the predicted label and its probability
            predicted_label = np.argmax(predicted_probabilities, axis=1)  # Index of highest probability
            predicted_emotion = loaded_label_encoder.inverse_transform(predicted_label)
            print("Predicted Emotion:", predicted_emotion[0])

            # Display probabilities for all classes
            emotion_probabilities = {emotion: prob for emotion, prob in zip(loaded_label_encoder.classes_, predicted_probabilities[0])}
            print("Probabilities for each emotion:")
            for emotion, probability in emotion_probabilities.items():
                print(f"{emotion}: {probability:.4f}")


        else:
            print("Model file not found. Train and save the model first.")
    else:
        print("Image capture failed.")
