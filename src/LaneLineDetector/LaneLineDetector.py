"""LaneLineDetector"""
from LaneLineDetector.Binarizer import Binarizer
from LaneLineDetector.CameraCalibrator import CameraCalibrator
from LaneLineDetector.Lane import Lane
from LaneLineDetector.Visualizer import Visualizer
from LaneLineDetector.Warper import Warper


class LaneLineDetector():
    """lane line Detector"""

    def __init__(self, frame_size=(720, 1280)):
        """Initialize lane line detector"""
        self.calibrator = CameraCalibrator()
        self.binarizer = Binarizer()
        self.warper = Warper()
        self.visualizer = Visualizer(self.warper)
        self.lane = Lane(image_size=frame_size)

        self.calibrator.loadParameters("./src/LaneLineDetector/calibration_coefficients.p")

    def process_frame(self, frame):
        """Process a video frame"""
        undist = self.calibrator.undistort(frame)
        binarized = self.binarizer.process(undist)
        warped = self.warper.warp(binarized)

        self.lane.detect_lane(warped)

        self.visualizer.draw_text_info(undist, self.lane.center_curvature, self.lane.center_offset)
        result = self.visualizer.draw_lane_on_road(undist, self.lane)

        return result
