import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score


# Load the uploaded .csv file to inspect its content
file_path = 'outputs/aus_data.csv'
data = pd.read_csv(file_path)

# Display the first few rows of the dataset to understand its structure
data.head()

# Extract the emotion label from the `image_path`
data['emotion'] = data['image_path'].str.split('/').str[-2]

# Display unique emotions to ensure proper extraction
unique_emotions = data['emotion'].unique()
unique_emotions

# Encode emotion labels into numerical format
label_encoder = LabelEncoder()
data['emotion_encoded'] = label_encoder.fit_transform(data['emotion'])

# Select features (valence, arousal, and AUxx columns) and target (emotion_encoded)
feature_columns = ['valence', 'arousal'] + [col for col in data.columns if col.startswith('AU')]
X = data[feature_columns]
y = data['emotion_encoded']

# Normalize the features
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

# Split the data into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

# Display the shapes of the resulting datasets
X_train.shape, X_test.shape, y_train.shape, y_test.shape

# Initialize the Random Forest Classifier
rf_model = RandomForestClassifier(random_state=42)

# Perform cross-validation to evaluate the model
cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='accuracy')

print("Cross-Validation Scores:", cv_scores)
print("Mean CV Accuracy:", cv_scores.mean())

# Hyperparameter Tuning using GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(
    estimator=rf_model,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    verbose=2,
    n_jobs=-1
)

# Fit GridSearchCV
grid_search.fit(X_train, y_train)

# Get the best parameters and evaluate on the test set
best_rf_model = grid_search.best_estimator_
y_pred = best_rf_model.predict(X_test)

print("Best Parameters:", grid_search.best_params_)
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
print("Accuracy:", accuracy_score(y_test, y_pred))
