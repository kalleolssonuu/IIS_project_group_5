import cv2
import os
from feat import Detector
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler as scaler
from sklearn.preprocessing import LabelEncoder as label_encoder

# Capture an image from the laptop camera
def capture_image():
    camera = cv2.VideoCapture(0)
    cv2.namedWindow("Capture Image")

    print("Press SPACE to capture the image.")
    while True:
        ret, frame = camera.read()
        if not ret:
            print("Failed to capture frame.")
            break
        cv2.imshow("Capture Image", frame)
        key = cv2.waitKey(1)
        if key % 256 == 32:  # Space key
            # Save captured image
            image_path = "captured_image.jpg"
            cv2.imwrite(image_path, frame)
            print(f"Image saved to {image_path}")
            break

    camera.release()
    cv2.destroyAllWindows()
    return image_path


# Main process
if __name__ == "__main__":
    # Step 1: Capture an image
    image_path = capture_image()