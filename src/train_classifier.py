"""Script to train a classifier to detect vehicles"""
import cv2
import glob
import matplotlib.image as mpimg
import numpy as np
import pickle
import time

from FeatureExtractor import FeatureExtractor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

# =============================================================================
# Initialize variables
# =============================================================================
feature_extractor = FeatureExtractor()


# =============================================================================
# Read in training images
# =============================================================================
notcar_images = glob.glob('./training_images/non-vehicles/**/*.png')
notcars = []
for image in notcar_images:
    notcars.append(image)


car_images = glob.glob('./training_images/vehicles/**/*.png')
cars = []
for image in car_images:
    cars.append(image)


# =============================================================================
# Define feature extraction function
# =============================================================================
def extract_features(imgs):
    """Extract features for given list of images"""
    features = []

    for img_name in imgs:
        image = cv2.imread(img_name)
        img_features = feature_extractor.get_features(image)
        features.append(img_features)

    return features


# =============================================================================
# Extract image features for sample_size images
# =============================================================================
print('Extracting features...')
sample_size = 8500
cars = cars[0:sample_size]
notcars = notcars[0:sample_size]

#cars = cars[0::2]
#notcars = notcars[0::2]

car_features = extract_features(cars)
notcar_features = extract_features(notcars)

X = np.vstack((car_features, notcar_features)).astype(np.float64)

X_scaler = StandardScaler().fit(X)
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)


# =============================================================================
# Train a linear classifier with the extracted feature sets
# =============================================================================
print('Train classifier...')
clf = LinearSVC()

t_start = time.time()
clf.fit(X_train, y_train)
t_stop = time.time()

print(round(t_stop - t_start, 2), 'Seconds to train SVC...')
print('Test Accuracy of SVC = ', round(clf.score(X_test, y_test), 4))


# =============================================================================
# Save classifier and scaler into pickle file for later usage
# =============================================================================
save_file = 'trained_classifier.p'
data = {
    'classifier': clf,
    'scaler': X_scaler
}

pickle.dump(data, open(save_file, 'wb'))

print('Saved trained model into ' + save_file)
