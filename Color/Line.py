# import sys
# import math
# import cv2 as cv
# import numpy as np

# def main(argv):
    
#     # default_file = 'sudoku.png'
#     # filename = argv[0] if len(argv) > 0 else default_file
#     # # Loads an image
#     # src = cv.imread(cv.samples.findFile(filename), cv.IMREAD_GRAYSCALE)
#     frame = cv.imread("./Color/uraban_frame.png")
#     # # Check if image is loaded fine
#     # if src is None:
#     #     print ('Error opening image!')
#     #     print ('Usage: hough_lines.py [image_name -- default ' + default_file + '] \n')
#     #     return -1


#     hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

#     src = cv.inRange(hsv_frame, np.array([0,0,255]), np.array([0,0,255]))
    
    
#     dst = cv.Canny(src, 50, 200, None, 3)
    
#     # Copy edges to the images that will display the results in BGR
#     cdst = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)
#     cdstP = np.copy(cdst)
    
#     lines = cv.HoughLines(dst, 1, np.pi / 180, 150, None, 0, 0)
    
#     if lines is not None:
#         for i in range(0, len(lines)):
#             rho = lines[i][0][0]
#             theta = lines[i][0][1]
#             a = math.cos(theta)
#             b = math.sin(theta)
#             x0 = a * rho
#             y0 = b * rho
#             pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
#             pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
#             cv.line(cdst, pt1, pt2, (0,0,255), 3, cv.LINE_AA)
    
    
#     linesP = cv.HoughLinesP(dst, 1, np.pi / 180, 50, None, 50, 10)
    
#     if linesP is not None:
#         for i in range(0, len(linesP)):
#             l = linesP[i][0]
#             cv.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv.LINE_AA)
    
#     cv.imshow("Source", src)
#     cv.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst)
#     cv.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)
    
#     cv.waitKey()
#     return 0
    
# if __name__ == "__main__":
#     main(sys.argv[1:])



import cv2
import numpy as np

car_mask = np.load('./car_mask.npy')


frame = cv2.imread("./urban_orginal_frame_2.png")

# frame = cv2.resize(image, (0, 0), fx = 0.5, fy = 0.5)

hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
white_mask = cv2.inRange(frame, np.array([240,240,240]), np.array([255,255,255])) * (1-car_mask)

side_mask = cv2.inRange(frame, np.array([130,0,108]), np.array([160,160,150])) * (1-car_mask)

red_mask = cv2.inRange(hsv_frame, np.array([140,70,0]), np.array([255,255,255])) * (1-car_mask)

cv2.putText(side_mask, "side_line", (0,140), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, 255, 2)
side_mask = cv2.rectangle(side_mask,(0,150),(256,190),255,1)


# cv2.imshow("hsv_frame", hsv_frame)
# cv2.imshow("white_mask", white_mask)
cv2.imshow("side_mask", side_mask)
# cv2.imshow("red_mask", red_mask)
cv2.waitKey(0)