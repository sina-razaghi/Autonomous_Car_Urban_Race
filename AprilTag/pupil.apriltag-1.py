import cv2 as cv
from pupil_apriltags import Detector
import time

def main():
   
    # cap = cv.VideoCapture('./AprilTag/dash_line-1.png')
    cap = cv.VideoCapture('./AprilTag/urban_frame2.png')
    
    # Detector
    at_detector = Detector(
        families='tag36h11',
        nthreads=1,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
    )

    while cap.isOpened():

        ret, image = cap.read()
        if not ret:
            break

        debug_image = image
        
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        tags = at_detector.detect(image, estimate_tag_pose=False, camera_params=None, tag_size=None,
        )
    
        #################################################################
        debug_image = draw_tags(debug_image, tags)
        
        cv.imshow('AprilTag Detect Demo', debug_image)

        cv.waitKey(1)

        time.sleep(30)



    cap.release()
    cv.destroyAllWindows()

def draw_tags(
    image,
    tags,
):
    for tag in tags:
        tag_family = tag.tag_family
        tag_id = tag.tag_id
        center = tag.center
        corners = tag.corners

        center = (int(center[0]), int(center[1]))
        corner_01 = (int(corners[0][0]), int(corners[0][1]))
        corner_02 = (int(corners[1][0]), int(corners[1][1]))
        corner_03 = (int(corners[2][0]), int(corners[2][1]))
        corner_04 = (int(corners[3][0]), int(corners[3][1]))

        # draw circle in the center
        cv.circle(image, (center[0], center[1]), 5, (0, 0, 255), 2)

        cv.line(image, (corner_01[0], corner_01[1]),(corner_02[0], corner_02[1]), (255, 0, 0), 2)
        cv.line(image, (corner_02[0], corner_02[1]),(corner_03[0], corner_03[1]), (255, 0, 0), 2)
        cv.line(image, (corner_03[0], corner_03[1]),(corner_04[0], corner_04[1]), (0, 255, 0), 2)
        cv.line(image, (corner_04[0], corner_04[1]),(corner_01[0], corner_01[1]), (0, 255, 0), 2)

        cv.putText(image, str(tag_id), (center[0] - 10, center[1] - 10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv.LINE_AA)

    return image

if __name__ == '__main__':
    main()