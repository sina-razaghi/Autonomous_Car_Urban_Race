import cv2
import numpy as np
import time
from tensorflow.keras.models import load_model

sign_model = load_model('best_model.h5')

def detect_lines(image):
    # tuning min_threshold, minLineLength, maxLineGap is a trial and error process by hand
    rho = 1  # precision in pixel, i.e. 1 pixel
    angle = np.pi / 180  # degree in radian, i.e. 1 degree
    min_threshold = 10  # minimal of votes
    lines = cv2.HoughLinesP(image, rho, angle, min_threshold, np.array([]), minLineLength=8,
                                    maxLineGap=4)
            
    return lines

def mean_lines(frame, lines):
    a = np.zeros_like(frame)
    try:
        left_line_x = []
        left_line_y = []
        right_line_x = []
        right_line_y = []
        for line in lines:
            for x1, y1, x2, y2 in line:
                slope = (y2 - y1) / (x2 - x1) # <-- Calculating the slope.
                if abs(slope) < 0.5: # <-- Only consider extreme slope
                    continue
                if slope <= 0: # <-- If the slope is negative, left group.
                    left_line_x.extend([x1, x2])
                    left_line_y.extend([y1, y2])
                else: # <-- Otherwise, right group.
                    right_line_x.extend([x1, x2])
                    right_line_y.extend([y1, y2])
        min_y = int(frame.shape[0] * (3 / 5)) # <-- Just below the horizon
        max_y = int(frame.shape[0]) # <-- The bottom of the image
        poly_left = np.poly1d(np.polyfit(
            left_line_y,
            left_line_x,
            deg=1
        ))
        left_x_start = int(poly_left(max_y))
        left_x_end = int(poly_left(min_y))
        poly_right = np.poly1d(np.polyfit(
            right_line_y,
            right_line_x,
            deg=1
        ))
        right_x_start = int(poly_right(max_y))
        right_x_end = int(poly_right(min_y))
        cv2.line(a, (left_x_start, max_y), (left_x_end, min_y), [255,255,0], 5)
        cv2.line(a, (right_x_start, max_y), (right_x_end, min_y), [255,255,0], 5)
        current_pix = (left_x_end+right_x_end)/2
    except:
        current_pix = 128
    return a, current_pix

def region_of_interest(image):
    (height, width) = image.shape
    mask = np.zeros_like(image)
    polygon = np.array([[
        (0, height),
        (0, 180),
        (80, 130),
        (256-80,130),
        (width, 180),   
        (width, height),
    ]], np.int32)

    cv2.fillPoly(mask, polygon, 255)
    masked_image = image * (mask)
    masked_image[:170,:]=0
    return masked_image

def horiz_lines(roi):
    # roi = mask[160:180, 96:160]
    try:
        lines = detect_lines(roi)
        lines = lines.reshape(-1,2,2)
        slope = (lines[:,1,1]-lines[:,0,1]) / (lines[:,1,0]-lines[:,0,0])
        
        if (lines[np.where(abs(slope)<0.2)]).shape[0] != 0:
            detected = True
        else:
            detected = False
    except:
        detected = False
    return detected

def turn_where(mask):
    roi = mask
    # cv2.imshow('turn where', roi)
    lines = detect_lines(roi)
    lines = lines.reshape(-1,2,2)
    slope = (lines[:,1,1]-lines[:,0,1]) / (lines[:,1,0]-lines[:,0,0])
    mean_pix = np.mean(lines[np.where(abs(slope)<0.2)][:,:,0])
    return mean_pix


def detect_side(side_mask):
    side_pix = np.mean(np.where(side_mask>0), axis=1)[1]
    return side_pix

def detect_sign(frame, hsv_frame):
    types = ['left', 'straight', 'right']
    mask = cv2.inRange(hsv_frame, np.array([100,160,90]), np.array([160,220,220]))
    mask[:30,:]=0
    try:
        points, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        sorted_points = sorted(points, key=len)

        if cv2.contourArea(sorted_points[-1])>30:
            x,y,w,h = cv2.boundingRect(sorted_points[-1])
            if (x>5) and (x+w<251) and (y>5) and (y+h<251):
                sign = frame[y:y+h,x:x+w]   
                sign = cv2.resize(sign, (25,25))/255
                frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
                return types[np.argmax(sign_model.predict(sign.reshape(1,25,25,3)))]
            else:
                return 'nothing' 
        else:
            return 'nothing'
    except:
        return 'nothing'
    

from pupil_apriltags import Detector

def detect_sign(frame):
    image = frame
    
    # Detector
    at_detector = Detector(
        # families='tag36h11',
        # nthreads=1,
        # quad_decimate=1.0,
        # quad_sigma=0.0,
        # refine_edges=1,
        # decode_sharpening=0.25,
    )

    

    debug_image = image
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    tags = at_detector.detect(image, estimate_tag_pose=False, camera_params=None, tag_size=None)
    for tag in tags:
        id = tag.tag_id
        print(f"tag id => {id}")
    #################################################################
        draw_tags(debug_image, tag)
    
    types = ['right', 'straight', 'left']
    
    if tags == []: 
        return 'nothing'
    else:
        return types[id]

    


def draw_tags(image, tag):
    # for tag in tags:
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
    cv2.circle(image, (center[0], center[1]), 5, (0, 0, 255), 2)

    cv2.line(image, (corner_01[0], corner_01[1]),(corner_02[0], corner_02[1]), (255, 0, 0), 2)
    cv2.line(image, (corner_02[0], corner_02[1]),(corner_03[0], corner_03[1]), (255, 0, 0), 2)
    cv2.line(image, (corner_03[0], corner_03[1]),(corner_04[0], corner_04[1]), (0, 255, 0), 2)
    cv2.line(image, (corner_04[0], corner_04[1]),(corner_01[0], corner_01[1]), (0, 255, 0), 2)

    cv2.putText(image, str(tag_id), (center[0] - 10, center[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow('AprilTag Detect Demo', image)
    cv2.waitKey(1)






def red_sign_state(red_mask):
    points, _ = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    sorted_points = sorted(points, key=len)
    try:
        red_area = cv2.contourArea(sorted_points[-1])
        if red_area > 50:
            print('red sign detected!')
            return True
        else:
            return False
    except:
        return False
    
def stop_the_car(car):
    car.setSteering(0)
    while car.getSpeed()>0:
        car.setSpeed(-100)
        car.getData()
    car.setSpeed(0)
    return True

def turn_the_car(car,s,t):
    time1 = time.time()
    while((time.time()-time1)<t):
        car.getData()
        car.setSteering(s)
        car.setSpeed(15)

def go_back(car, t):
    time1 = time.time()
    while((time.time()-time1)<t):
        car.getData()
        car.setSpeed(-15)
    car.setSpeed(0)
