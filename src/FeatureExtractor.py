"""Feature Extractor class"""
import cv2
import numpy as np
from skimage.feature import hog


class FeatureExtractor():
    """Feature Extractor"""

    def __init__(self):
        """Initialize feature extractor"""
        # Set hog extractor parameters
        self.hog_orientations = 9
        self.pixel_per_cell = 16
        self.cells_per_block = 2
        self.spatial_size = (32, 32)
        self.histogram_bins = 32

    def bin_spatial(self, img):
        """Resize given image to size and convert to a single horizontal array"""
        color1 = cv2.resize(img[:, :, 0], self.spatial_size).ravel()
        color2 = cv2.resize(img[:, :, 1], self.spatial_size).ravel()
        color3 = cv2.resize(img[:, :, 2], self.spatial_size).ravel()
        return color1
        return np.hstack((color1, color2, color3))

    def color_hist(self, img):
        """Compute color channel histograms and combine into single array."""
        # Compute the histogram of the color channels separately
        channel1_hist = np.histogram(img[:, :, 0], bins=self.histogram_bins)
        channel2_hist = np.histogram(img[:, :, 1], bins=self.histogram_bins)
        channel3_hist = np.histogram(img[:, :, 2], bins=self.histogram_bins)
        # Concatenate the histograms into a single feature vector
        hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
        # Return the individual histograms, bin_centers and feature vector
        return hist_features

    def get_hog_features(self, img, vis=False, feature_vec=True):
        """Extract hog features for given image."""
        # Call with two outputs if vis==True
        if vis:
            features, hog_image = hog(img, orientations=self.hog_orientations,
                                      pixels_per_cell=(self.pixel_per_cell, self.pixel_per_cell),
                                      cells_per_block=(self.cells_per_block, self.cells_per_block),
                                      transform_sqrt=False,
                                      visualise=vis, feature_vector=feature_vec)
            return features, hog_image
        # Otherwise call with one output
        else:
            features = hog(img, orientations=self.hog_orientations,
                           pixels_per_cell=(self.pixel_per_cell, self.pixel_per_cell),
                           cells_per_block=(self.cells_per_block, self.cells_per_block),
                           transform_sqrt=False,
                           visualise=vis, feature_vector=feature_vec)
        return features
