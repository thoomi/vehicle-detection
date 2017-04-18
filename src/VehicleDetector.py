"""Vehicle Detector"""
import cv2
import numpy as np
import pickle

from BufferedHeatmap import BufferedHeatmap
from FeatureExtractor import FeatureExtractor


class VehicleDetector():
    """Vehicle Detector"""

    def __init__(self, frame_size=(720, 1280)):
        """Initialize vehicle detector"""
        # Load scaler and classifier from pickle file
        loaded_data = pickle.load(open('trained_classifier.p', 'rb'))
        self.scaler = loaded_data['scaler']
        self.classifier = loaded_data['classifier']

        self.feature_extractor = FeatureExtractor()

        # Define upper and lower limit of the main search window
        self.lower_window_limit = 400
        self.upper_window_limit = 680

        self.heatmap = BufferedHeatmap(input_size=frame_size, buffer_size=24, threshold=8)

        self.detection_areas = []
        self.detection_areas.append(([400, 528], 48))
        self.detection_areas.append(([400, 528], 64))
        self.detection_areas.append(([400, 656], 128))
        #self.detection_areas.append(([360, 720], 160))

    def process_frame(self, frame):
        """Process a video frame"""
        draw_img = np.copy(frame)

        # Generate detection windows
        windows = []
        for area in self.detection_areas:
            new_windows = self.slide_window(draw_img, x_start_stop=[None, None], y_start_stop=area[0],
                        xy_window=(area[1], area[1]), xy_overlap=(0.7, 0.7))

            windows.extend(new_windows)

        # Run detection on each window
        detections = []
        for window in windows:
            img_patch = cv2.resize(frame[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
            features = self.feature_extractor.get_features(img_patch)
            features = self.scaler.transform(features.reshape(1, -1))
            prediction = self.classifier.predict(features)

            if prediction == 1:
                detections.append(window)
                #cv2.rectangle(draw_img, window[0], window[1], (0, 0, 255), 6)

        self.heatmap.add_heat(detections)

        labels = self.heatmap.get_labels()

        self.draw_labeled_bboxes(draw_img, labels)

        heatmap = self.heatmap.get_heatmap()
        mini_heatmap = cv2.resize(heatmap, (350, 200), None, 0, 0, cv2.INTER_LINEAR)
        mini_heatmap *= 255
        mini_heatmap = np.dstack([mini_heatmap, mini_heatmap, mini_heatmap])
        draw_img[0:mini_heatmap.shape[0], 0:mini_heatmap.shape[1]] = mini_heatmap


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

        self.heatmap.add_heat(bboxes)

        return bboxes

    def slide_window(self, img, x_start_stop=[None, None], y_start_stop=[None, None],
                     xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
        # If x and/or y start/stop positions not defined, set to image size
        if x_start_stop[0] == None:
            x_start_stop[0] = 0
        if x_start_stop[1] == None:
            x_start_stop[1] = img.shape[1]
        if y_start_stop[0] == None:
            y_start_stop[0] = 0
        if y_start_stop[1] == None:
            y_start_stop[1] = img.shape[0]
        # Compute the span of the region to be searched
        xspan = x_start_stop[1] - x_start_stop[0]
        yspan = y_start_stop[1] - y_start_stop[0]
        # Compute the number of pixels per step in x/y
        nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
        ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
        # Compute the number of windows in x/y
        nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
        ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
        nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step)
        ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step)
        # Initialize a list to append window positions to
        window_list = []
        # Loop through finding x and y window positions
        # Note: you could vectorize this step, but in practice
        # you'll be considering windows one by one with your
        # classifier, so looping makes sense
        for ys in range(ny_windows):
            for xs in range(nx_windows):
                # Calculate window position
                startx = xs*nx_pix_per_step + x_start_stop[0]
                endx = startx + xy_window[0]
                starty = ys*ny_pix_per_step + y_start_stop[0]
                endy = starty + xy_window[1]

                # Append window position to list
                window_list.append(((startx, starty), (endx, endy)))
        # Return the list of windows
        return window_list

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
