"""Combined Lane Line and Vehicle detection pipeline"""
from moviepy.editor import VideoFileClip
from optparse import OptionParser

from LaneLineDetector.LaneLineDetector import LaneLineDetector
from VehicleDetector.VehicleDetector import VehicleDetector

# =============================================================================
# Get command line arguments
# =============================================================================
parser = OptionParser()
(options, args) = parser.parse_args()

input_video_name = args[0]
output_video_name = 'result_' + input_video_name

# =============================================================================
# Create processing instances
# =============================================================================
lane_line_detector = LaneLineDetector()
vehicle_detector = VehicleDetector()


# =============================================================================
# Preprocessing pipeline
# =============================================================================
def process_image(image):
    """Process a single image"""
    result = lane_line_detector.process_frame(image)
    vehicle_labels = vehicle_detector.process_frame(image, output_labels=True)

    vehicle_detector.draw_labeled_bboxes(result, vehicle_labels)

    return result


# image = cv2.imread('./test_images/test6.jpg')
# output = process_image(image)
# plt.imshow(output)
# plt.show()

# =============================================================================
# Process video file
# =============================================================================
clip1 = VideoFileClip(input_video_name)
project_clip = clip1.fl_image(process_image)
project_clip.write_videofile(output_video_name, audio=False)
