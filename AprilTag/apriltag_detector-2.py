import cv2
from pupil_apriltags import Detector

# Load the imag
# image = cv2.imread('./AprilTag/urban_frame4.png')
image = cv2.imread('./AprilTag/dash_line-1.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Create a detector object
detector = Detector(
        families='tag36h11',
        nthreads=1,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25
        )

# Detect the tags in the image
tags = detector.detect(gray)
#print(type(tags))

# Print the tag information
for tag in tags:
  print("Tag ID: {}".format(tag.tag_id))
  print("Tag center: {}".format(tag.center))
  print("Tag corners: {}".format(tag.corners))
  print('tag family: {}'.format(tag.tag_family))

# Show the image with the detected tags
cv2.imshow("AprilTag Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
