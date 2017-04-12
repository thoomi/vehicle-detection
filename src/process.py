"""CarND Vehicle Detection"""
import cv2
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
from optparse import OptionParser
from VehicleDetector import VehicleDetector

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
vehicle_detector = VehicleDetector()


# =============================================================================
# Preprocessing pipeline
# =============================================================================
def process_image(image):
    """Process a single image"""
    result = vehicle_detector.process_frame(image)

    return result


# image = cv2.imread('./test_images/test1.jpg')
# output = process_image(image)
# plt.imshow(output[..., ::-1])
# plt.show()

# =============================================================================
# Process video file
# =============================================================================
clip1 = VideoFileClip(input_video_name)
project_clip = clip1.fl_image(process_image)
project_clip.write_videofile(output_video_name, audio=False)
