# Using known shape size (width of tape) and scaling (plus some trig probably) to find distance
# center the reflective tape in the camera fov
# measure the vertical angle from cam to tape

# We have seen that the retro-reflective tape will not shine unless a light source
# is directed at it, and the light source must pass very near the camera lens or the observer’s eyes
# Thus, we use a ring light bc they work quite well for causing retro-reflective tape to shine

import math
import imutils
import numpy as np
import cv2
import glob



#pitch_angle = math.degrees(math.atan((y_cam_coord - 243.65759666) / config.focal_len))
#yaw_angle = math.degrees(math.atan((x_cam_coord - 293.51663222) / config.focal_len))

    # Next step is to use thresholding. Thresholding is taking an image, and throwing away any pixels
    # that aren’t in a specific color range. The result of thresholding is generally a one-dimensional
    # image in which a pixel is either “on” or “off. We'll use HSV to specify the color of the target
    ##gray = cv2.cvtColor(cFrame, cv2.COLOR_BGR2GRAY)
    ##blurred = cv2.blur(gray, (4, 4))

import cv2
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
cap.set(cv2.CAP_PROP_EXPOSURE,-4)


def nothing(x):
    pass


cv2.namedWindow("HSV Adjustments")

cv2.createTrackbar("Lower_H", "HSV Adjustments", 0, 255, nothing)
cv2.createTrackbar("Lower_S", "HSV Adjustments", 0, 255, nothing)
cv2.createTrackbar("Lower_V", "HSV Adjustments", 0, 255, nothing)
cv2.createTrackbar("Upper_H", "HSV Adjustments", 0, 255, nothing)
cv2.createTrackbar("Upper_S", "HSV Adjustments", 0, 255, nothing)
cv2.createTrackbar("Upper_V", "HSV Adjustments", 0, 255, nothing)


while True:

    ret, frame = cap.read()

    cv2.imshow('testing!', frame)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    l_h = cv2.getTrackbarPos("Lower_H", "HSV Adjustments")
    l_s = cv2.getTrackbarPos("Lower_S", "HSV Adjustments")
    l_v = cv2.getTrackbarPos("Lower_V", "HSV Adjustments")
    u_h = cv2.getTrackbarPos("Upper_H", "HSV Adjustments")
    u_s = cv2.getTrackbarPos("Upper_S", "HSV Adjustments")
    u_v = cv2.getTrackbarPos("Upper_V", "HSV Adjustments")

    lower_bound = np.array([l_h, l_s, l_v])
    upper_bound = np.array([u_h, u_s, u_v])

    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    result = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow('Original', frame)
    cv2.imshow('HSV', hsv)
    cv2.imshow('Mask', mask)
    cv2.imshow('Result', result)

    key = cv2.waitKey(1)
    if key == 27:
        print("l_h:" + str(l_h) + " " + "l_s:" + str(l_s) + " " + "l_v:" + str(l_v) + " " +
              "u_h: " + str(u_h) + " " + "u_s:" + str(u_s) + " " + "u_v:" + " " + str(u_v))
        break

cap.release()
cv2.destroyAllWindows()
