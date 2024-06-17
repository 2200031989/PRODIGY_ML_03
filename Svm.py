import os
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from matplotlib import image as mpimg
from tqdm import tqdm

# Directories
train_dir = 'train1'  # Replace with actual path to your training data
test_dir = 'test2'  # Replace with actual path to your test data

# Image size for resizing
image_size = (64, 64)


def load_images_from_folder(folder):
    images = []
    labels = []
    filenames = os.listdir(folder)
    for filename in tqdm(filenames, desc=f'Loading images from {folder}'):
        if 'cat' in filename:
            label = 0  # 'cat'
        elif 'dog' in filename:
            label = 1  # 'dog'
        else:
            continue
        img_path = os.path.join(folder, filename)
        img = mpimg.imread(img_path)
        img = np.array(img)
        img = resize_image(img, image_size)
        images.append(img)
        labels.append(label)
    print(f'Loaded {len(images)} images and {len(labels)} labels from {folder}')
    print(f'Labels: {labels}')
    return np.array(images), np.array(labels)


def resize_image(img, size):
    return np.resize(img, size)


def extract_features(images):
    # Flatten the image arrays
    features = []
    for img in tqdm(images, desc='Extracting features'):
        features.append(img.flatten())
    return np.array(features)


# Load training data
train_images, train_labels = load_images_from_folder(train_dir)

# Check labels to ensure both classes are present
unique_labels = np.unique(train_labels)
if len(unique_labels) < 2:
    raise ValueError("Training data does not contain samples from both classes (cats and dogs).")

# Extract features
train_features = extract_features(train_images)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(train_features, train_labels, test_size=0.2, random_state=42)

# Create and train SVM model
clf = svm.SVC(kernel='linear')

# Train SVM model
clf.fit(X_train, y_train)

# Validate the model
y_val_pred = clf.predict(X_val)
validation_accuracy = accuracy_score(y_val, y_val_pred) * 100
print(f'Validation Accuracy: {validation_accuracy:.2f}%')

# Load test data
test_images, _ = load_images_from_folder(test_dir)

# Debug print to check if test_images is empty or not loaded
print(f'Loaded {len(test_images)} images from test directory')

# Check if test_images is empty or not loaded


# Extract features for test data
test_features = extract_features(test_images)

# Debug print to check if test_features is empty or not extracted
print(f'Extracted {len(test_features)} features from test images')

# Check if test_features is empty after extraction

# Predict using the trained model
test_predictions = []
for i, test_feature in enumerate(tqdm(test_features, desc='Predicting test images')):
    prediction = clf.predict([test_feature])[0]
    label = 'cat' if prediction == 0 else 'dog'
    test_predictions.append(label)
    print(f'{i + 1},{label}')  # Print in the desired format
