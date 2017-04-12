"""Vehicle Detector"""
import cv2
import numpy as np
import pickle

from FeatureExtractor import FeatureExtractor
from scipy.ndimage.measurements import label


class VehicleDetector():
    """Vehicle Detector"""

    def __init__(self):
        """Initialize vehicle detector"""
        # Load scaler and classifier from pickle file
        loaded_data = pickle.load(open('trained_classifier.p', 'rb'))
        self.scaler = loaded_data['scaler']
        self.classifier = loaded_data['classifier']

        self.feature_extractor = FeatureExtractor()

        # Define upper and lower limit of the main search window
        self.lower_window_limit = 400
        self.upper_window_limit = 680

        self.heatmap = None

    def add_heat(self, bbox_list):
        """Add heat within given bounding boxes"""
        for box in bbox_list:
            self.heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    def threshold_heat(self, threshold):
        """Apply a threshold to heatmap"""
        self.heatmap[self.heatmap <= threshold] = 0

    def draw_labeled_bboxes(self, img, labels):
        """Draw heatmap bounding boxes"""
        # Iterate through all detected cars
        for car_number in range(1, labels[1] + 1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            # Draw the box on the image
            cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
        # Return the image
        return img

    def process_frame(self, frame):
        """Process a video frame"""
        draw_img = np.copy(frame)

        # Initialize heatmap
        self.heatmap = np.zeros_like(frame[:, :, 0]).astype(np.float)

        # Process frame at multiple scales
        self.detect_vehicles(frame, 1)
        self.detect_vehicles(frame, 1.5)
        self.detect_vehicles(frame, 2)

        # Get labels based on heatmap and draw them
        self.threshold_heat(1)
        labels = label(self.heatmap)
        draw_img = self.draw_labeled_bboxes(draw_img, labels)

        # for box in bboxes:
        #     print(box[0])
        #     cv2.rectangle(draw_img, box[0], box[1], (0, 0, 255), 6)

        return draw_img

    def detect_vehicles(self, image, scale):
        """Detect vehicles whithin an image using a classifier"""
        image = image.astype(np.float32) / 255

        image_of_interest = image[self.lower_window_limit:self.upper_window_limit, :, :]
        image_converted = cv2.cvtColor(image_of_interest, cv2.COLOR_RGB2YCrCb)
        image_scaled = cv2.resize(image_converted,
                                  (np.int(image_converted.shape[1] / scale),
                                   np.int(image_converted.shape[0] / scale)))

        ch1 = image_scaled[:, :, 0]
        ch2 = image_scaled[:, :, 1]
        ch3 = image_scaled[:, :, 2]

        # Define blocks and steps
        nxblocks = (ch1.shape[1] // self.feature_extractor.pixel_per_cell) - 1
        nyblocks = (ch1.shape[0] // self.feature_extractor.pixel_per_cell) - 1
        # nfeat_per_block = self.hog_orientations * self.cell_per_block**2

        # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
        window = 64
        nblocks_per_window = (window // self.feature_extractor.pixel_per_cell) - 1
        cells_per_step = 2  # Instead of overlap, define how many cells to step
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step

        # Compute individual channel HOG features for the entire image
        hog1 = self.feature_extractor.get_hog_features(ch1, feature_vec=False)
        hog2 = self.feature_extractor.get_hog_features(ch2, feature_vec=False)
        hog3 = self.feature_extractor.get_hog_features(ch3, feature_vec=False)

        bboxes = []

        for xb in range(nxsteps):
            for yb in range(nysteps):
                xpos = xb * cells_per_step
                ypos = yb * cells_per_step

                # Extract HOG for this patch
                hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

                xleft = xpos * self.feature_extractor.pixel_per_cell
                ytop = ypos * self.feature_extractor.pixel_per_cell

                # Extract the image patch
                # subimg = cv2.resize(image_scaled[ytop:ytop + window, xleft:xleft + window], (64, 64))

                # Get color features
                # spatial_features = self.feature_extractor.bin_spatial(subimg)
                # hist_features = self.feature_extractor.color_hist(subimg)

                # Scale features and make a prediction
                # test_features = self.scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
                test_features = self.scaler.transform(hog_features.reshape(1, -1))

                # Use classifier to predict output for given features
                test_prediction = self.classifier.predict(test_features)
                # test_prediction = 1
                if test_prediction == 1:
                    xbox_left = np.int(xleft * scale)
                    ytop_draw = np.int(ytop * scale)
                    win_draw = np.int(window * scale)

                    top_left_corner = (xbox_left, ytop_draw + self.lower_window_limit)
                    bottom_right_corner = (xbox_left + win_draw, ytop_draw + win_draw + self.lower_window_limit)

                    bboxes.append([top_left_corner, bottom_right_corner])
                    # cv2.rectangle(draw_img, top_left_corner, bottom_right_corner, (0, 0, 255), 6)

        self.add_heat(bboxes)

        return bboxes
