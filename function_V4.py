import cv2
import numpy as np
import time
from pupil_apriltags import Detector

detector = Detector(
        families='tag36h11',
        nthreads=1,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
        debug=0
        )


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


def detect_sign_apriltag(image):
    types = ['nothing', 'straight', 'right', 'left', 'straight', 'stop']
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    tags = detector.detect(gray, estimate_tag_pose=False, camera_params=None, tag_size=None)
    id = 0

    for tag in tags:
        id = tag.tag_id
        # try:
        #     print(f"Tag ID: {tag.tag_id} => {types[id]}")
        # except:
        #     print(f"Tag ID: {tag.tag_id} => Others")
        #     id == 0
        # print("Tag center: {}".format(tag.center))
        # print("Tag corners: {}".format(tag.corners))
        # print('tag family: {}'.format(tag.tag_family))

    if id == 0:
        return types[id]
    else:
        return types[id]



def red_sign_state(red_mask):
    points, _ = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    sorted_points = sorted(points, key=len)
    try:
        red_area = cv2.contourArea(sorted_points[-1])
        if red_area > 50:
            print(f"red_area_size => {red_area}")
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
