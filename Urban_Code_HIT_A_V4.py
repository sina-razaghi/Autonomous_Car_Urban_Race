import AVISEngineNew
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
import function_V3
from os import system
np.seterr(all="ignore")

car = AVISEngineNew.Car()
car.connect("127.0.0.1", 25001)

# variables
REFRENCE = 128
CURRENT_PXL = 128

sign_state = 'nothing'
position = 'right'

kp = 2
ki = 0.1
kd = 0.1
steer = 0
sensors = [1500,1500,1500]
# car_mask = cv2.imread('./car_mask.jpg',0)
# car_mask[car_mask<128] = 0
# car_mask[car_mask>=128] = 1
car_mask = np.load('./car_mask.npy')

# initializing
for _ in range(10):
    car.setSteering(0)
    car.setSpeed(10)
    car.getData()

# main loop
while(True):  
    system('cls')
    # getting data 
    car.getData()
    sensors = car.getSensors() 
    image = car.getImage()

    frame = cv2.resize(image, (0, 0), fx = 0.5, fy = 0.5)

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    white_mask = cv2.inRange(frame, np.array([240,240,240]), np.array([255,255,255])) * (1-car_mask)
        
    red_mask = cv2.inRange(hsv_frame, np.array([140,70,0]), np.array([255,255,255])) * (1-car_mask)
    
    # vertical lines
    lines = function_V3.detect_lines(function_V3.region_of_interest(white_mask))
    two_line_mask, CURRENT_PXL = function_V3.mean_lines(white_mask, lines)

    # white horiz line
    horiz_detected = function_V3.horiz_lines(white_mask[160:180, 96:160])
    horiz_detected_90 = function_V3.horiz_lines(white_mask[130:140, 96:160])

    # only on turns and obstacle
    mean_pix = function_V3.turn_where(white_mask)
    side_pix = function_V3.detect_side(white_mask[155:256, :])
    
    # detecting sign type
    red_sign = function_V3.red_sign_state(red_mask)

    sign = function_V3.detect_sign_apriltag(image[100:400, 280:512])

    if sign == 'left':
        sign_state = 'left'
    elif sign == 'straight':
        sign_state = 'straight'
    elif sign == 'right':
        sign_state = 'right'
    elif sign == 'stop':
        sign_state = 'stop'


    error = REFRENCE - CURRENT_PXL 
    steer = -(kp * error)
    car.setSteering(int(steer))
    car.setSpeed(30)

    if side_pix >= 128:
        position = 'right'
    elif side_pix < 128 and side_pix > 0:
        position = 'left'
    print(f"Position => {position} | {side_pix}")

    if not red_sign:
        print(f"Movment => {sign_state}")
        if sign_state == 'nothing':
            if horiz_detected:
                car.setSteering(0)
                ret = function_V3.stop_the_car(car)
                if mean_pix < 128:
                    if position == 'right':
                        function_V3.go_back(car,4.5)
                        function_V3.turn_the_car(car,-100,10.5)
                    else:
                        function_V3.turn_the_car(car,-80,8)
                else:
                    if position == 'left':
                        function_V3.go_back(car,8)
                    else:
                        function_V3.go_back(car,4.5)
                    function_V3.turn_the_car(car,100,10)

            elif horiz_detected_90:
                # turn based on mean_pix
                car.setSteering(0)
                ret = function_V3.stop_the_car(car)
                mean_pix = function_V3.turn_where(white_mask)
                print('mean_pix :', mean_pix)

                if mean_pix < 128:
                    # if side_pix > 128:
                    #     function.turn_the_car(car,-100,8.5)
                    # else:
                    #     function.turn_the_car(car,0,5)
                    #     function.turn_the_car(car,-100,6)
                    function_V3.turn_the_car(car,100,2)
                    function_V3.turn_the_car(car,-100,8.3)
                else:
                    # if side_pix > 128:
                    #     function.turn_the_car(car,-100,1.3)
                    #     function.turn_the_car(car,100,8.5)
                    # else:
                    #     # function.go_back(car,5)
                    #     function.turn_the_car(car,100,11)
                    function_V3.turn_the_car(car,-100,2)
                    function_V3.turn_the_car(car,100,8.3)
            
        elif sign_state == 'left':
            if horiz_detected:
                car.setSteering(0)
                ret = function_V3.stop_the_car(car)
                time.sleep(3)
                
                if position == 'right':
                    function_V3.turn_the_car(car,-35,9.5)
                    function_V3.turn_the_car(car,-100,3)
                else:
                    function_V3.turn_the_car(car,0,6)
                    function_V3.turn_the_car(car,-100,6)

                sign_state = 'nothing'

        elif sign_state == 'straight':
            if horiz_detected:
                car.setSteering(0)
                ret = function_V3.stop_the_car(car)
                time.sleep(3)

                if position == 'right':
                    function_V3.turn_the_car(car,0,11)
                else:
                    function_V3.turn_the_car(car,0,3)
                    function_V3.turn_the_car(car,100,4)
                    function_V3.turn_the_car(car,-100,3.6)
                    function_V3.turn_the_car(car,0,3)                

                sign_state = 'nothing'

        elif sign_state == 'right':
            if horiz_detected:
                car.setSteering(0)
                ret = function_V3.stop_the_car(car)
                time.sleep(3)

                if position == 'right':
                    function_V3.turn_the_car(car,0,4.5)
                    function_V3.turn_the_car(car,100,6.5)
                else:
                    function_V3.turn_the_car(car,75,10.5)
                sign_state = 'nothing'

    
    else:
        print('red sign detected. stopping the car ...')
        if horiz_detected:
            car.setSteering(0)
            ret = function_V3.stop_the_car(car)
            time.sleep(1)
            break

    if 500 < sensors[1] and sensors[1] < 700:
        print('Obstacle')
        ret = function_V3.stop_the_car(car)
        print('side_pix :', side_pix)
        time.sleep(3)
        if position == 'right':
            function_V3.turn_the_car(car,-100,5.5)
            function_V3.turn_the_car(car,100,6.5)
            function_V3.turn_the_car(car,-100,2.5)
        else:
            function_V3.turn_the_car(car,100,7.5)
            function_V3.turn_the_car(car,-100,2.5)

    elif sensors[1] < 500:
        print('Obstacle so Close ...')
        ret = function_V3.stop_the_car(car)
        print('side_pix :', side_pix)
        time.sleep(3)
        if position == 'right':
            while car.getSensors()[1] < 500:
                car.getData() 
                function_V3.go_back(car, 0.2)
            car.setSpeed(0)
            function_V3.turn_the_car(car,-100,7.5)
            function_V3.turn_the_car(car,100,6.5)
            function_V3.turn_the_car(car,-100,2.5)
        else:
            while car.getSensors()[1] < 500:
                car.getData() 
                function_V3.go_back(car, 0.2)
            car.setSpeed(0)            
            function_V3.turn_the_car(car,100,7.5)
            function_V3.turn_the_car(car,-100,2.5)

    
    # showing some info

    print(f"State => {sign_state}")
    debugFrame = np.copy(frame)
    cv2.putText(debugFrame, "horiz_line", (96,195), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 2)
    debugFrame = cv2.rectangle(debugFrame,(96,160),(160,180),(0,255,0),1)
    cv2.putText(debugFrame, "side_line", (0,145), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 255), 2)
    debugFrame = cv2.rectangle(debugFrame,(0,155),(256,256),(0,255,255),1)
    cv2.putText(debugFrame, "horiz_90", (85,120), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 0), 2)
    debugFrame = cv2.rectangle(debugFrame,(96,130),(160,140),(255,255,0),1)

    # result = np.concatenate(
    #     [np.concatenate(
    #         [debugFrame,
    #          cv2.cvtColor(function_V3.region_of_interest(white_mask)*255, cv2.COLOR_GRAY2BGR)
    #         ], 1),
    #     np.concatenate(
    #         [cv2.cvtColor(two_line_mask, cv2.COLOR_GRAY2BGR),
    #          cv2.cvtColor(side_mask, cv2.COLOR_GRAY2BGR)
    #         ],0)
    #     ],1)

    result = np.concatenate(
        [np.concatenate(
            [debugFrame,
             cv2.cvtColor(function_V3.region_of_interest(white_mask)*255, cv2.COLOR_GRAY2BGR)
            ], 0),
            cv2.cvtColor(two_line_mask, cv2.COLOR_GRAY2BGR)
        ],0)

    cv2.imshow('car\'s perception', result)   

    key = cv2.waitKey(1)
    if key == ord('w'):
        frame = car.getImage()
        cv2.imwrite('./urban_orginal_frame.png', image)
