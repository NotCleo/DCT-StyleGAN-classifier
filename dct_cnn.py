import cv2
import os
import numpy as np
from mtcnn import MTCNN
from sklearn.model_selection import train_test_split
from tensorflow.keras import models, layers
import tensorflow as tf

def load_images_from_folder(folder):
    images = []
    labels = []
    for label in ['real', 'fake']:
        path = './dataset_real' if label == 'real' else './dataset_fake'
        for filename in os.listdir(path):
            img = cv2.imread(os.path.join(path, filename))
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Do not forget this cruial conversion to Grayscale ;)
                images.append(img)
                labels.append(0 if label == 'real' else 1)
    return images, labels

def detect_and_crop_faces(images):
    detector = MTCNN()
    cropped_faces = []
    for img in images:
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
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    float_img = np.float32(image)
    dct = cv2.dct(float_img)
    return dct[:32, :32]  # This is the good stuff
  
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
    return X, y

#Fun stuff starts here,


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
    data_dir = '....'  #provide the directory real/GAN here
    print("Loading images...")
    images, labels = load_images_from_folder(data_dir)
    
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
    
    test_image_path = '....'  #provide the path here
    threshold = 0.6  # manually adjust this threshold
    
    if os.path.exists(test_image_path):
        print(f"\nTesting on {test_image_path}...")
        img = cv2.imread(test_image_path)
        face = detect_and_crop_faces([img])[0]
        if face is not None:
            dct = compute_dct(face)
            dct = dct[np.newaxis, ..., np.newaxis]
            score = model.predict(dct)[0][0]
            prediction = "Fake" if score > threshold else "Real"
            print(f"Score: {score:.4f}, Prediction: {prediction} (Threshold: {threshold})")
        else:
            print("No face detected in test image.")
    else:
        print("Test image not found. Skipping inference.")

if __name__ == "__main__":
    main()
