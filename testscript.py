import cv2
import os
import numpy as np
from mtcnn import MTCNN
from sklearn.model_selection import train_test_split
from tensorflow.keras import models, layers
import tensorflow as tf
import random

def load_images_from_folder():
    real_path = "/home/amruth22/deepfake_detection_project/archive/Final Dataset/Real"
    fake_path = "/home/amruth22/deepfake_detection_project/archive/Final Dataset/Fake"
    
    images = []
    labels = []
    target_per_category = 500  
    offset = 500  
    if not os.path.exists(real_path):
        print(f"Real directory not found: {real_path}")
        return [], []
    
    real_files = [f for f in os.listdir(real_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if len(real_files) < (offset + target_per_category):
        print(f"Warning: Not enough real images found after offset")
        selected_real = real_files[offset:]
    else:
        real_files = real_files[offset:]  
        selected_real = random.sample(real_files, target_per_category)
    
    print(f"Loading {len(selected_real)} real images (starting after image {offset})...")
    for filename in selected_real:
        img = cv2.imread(os.path.join(real_path, filename))
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            images.append(img)
            labels.append(0) 
    if not os.path.exists(fake_path):
        print(f"Fake directory not found: {fake_path}")
        return [], []
    
    fake_files = [f for f in os.listdir(fake_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if len(fake_files) < (offset + target_per_category):
        print(f"Warning: Not enough fake images found after offset")
        selected_fake = fake_files[offset:]
    else:
        fake_files = fake_files[offset:]  # Skip first 500
        selected_fake = random.sample(fake_files, target_per_category)
    
    print(f"Loading {len(selected_fake)} fake images (starting after image {offset})...")
    for filename in selected_fake:
        img = cv2.imread(os.path.join(fake_path, filename))
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            images.append(img)
            labels.append(1)  
    print(f"Total images loaded: {len(images)} (Real: {labels.count(0)}, Fake: {labels.count(1)})")
    return images, labels

def detect_and_crop_faces(images):
    detector = MTCNN()
    cropped_faces = []
    for img in images:
        if len(img.shape) == 2:  # If grayscale
            rgb_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:  # If already RGB/BGR
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        results = detector.detect_faces(rgb_img)
        if results:
            x, y, w, h = results[0]['box']
            x, y = max(0, x), max(0, y)
            face = img[y:y+h, x:x+w]
            face = cv2.resize(face, (64, 64))
            cropped_faces.append(face)
        else:
            cropped_faces.append(None)
    return cropped_faces

def compute_dct(image):
    if image is None:
        return None
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    float_img = np.float32(image)
    dct = cv2.dct(float_img)
    return dct[:32, :32]  # Subset for efficiency

def prepare_data(cropped_faces, labels):
    X, y = [], []
    for face, label in zip(cropped_faces, labels):
        if face is not None:
            dct = compute_dct(face)
            if dct is not None:
                X.append(dct)
                y.append(label)
    X = np.array(X)[..., np.newaxis]  # (N, 32, 32, 1)
    y = np.array(y)
    print(f"Prepared {len(X)} samples for training after DCT processing")
    return X, y

def build_cnn():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def main():
    print("Loading images...")
    images, labels = load_images_from_folder()
    
    if not images:
        print("No images loaded. Exiting.")
        return
    
    print("Detecting and cropping faces...")
    cropped_faces = detect_and_crop_faces(images)
    
    print("Preparing DCT features...")
    X, y = prepare_data(cropped_faces, labels)
    if len(X) == 0:
        print("No valid data after preprocessing. Check your dataset.")
        return
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training CNN...")
    model = build_cnn()
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
    
    model.save('deepfake_detector_model.h5')
    print("Model saved as 'deepfake_detector_model.h5'")
    
    test_image_path = "/home/amruth22/deepfake_detection_project/archive/Final Dataset/Fake/001DDU0NI4.jpg"
    threshold = 0.6 
    
    if os.path.exists(test_image_path):
        print(f"\nTesting on {test_image_path}...")
        img = cv2.imread(test_image_path)
        if img is not None:
            face = detect_and_crop_faces([img])[0]
            if face is not None:
                dct = compute_dct(face)
                if dct is not None:
                    dct = dct[np.newaxis, ..., np.newaxis]
                    score = model.predict(dct)[0][0]
                    prediction = "Fake" if score > threshold else "Real"
                    print(f"Score: {score:.4f}, Prediction: {prediction} (Threshold: {threshold})")
                else:
                    print("Failed to compute DCT features.")
            else:
                print("No face detected in test image.")
        else:
            print("Failed to load test image.")
    else:
        print(f"Test image not found at {test_image_path}. Skipping inference.")

if __name__ == "__main__":
    main()
