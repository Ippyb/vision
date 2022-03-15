# desmos: https://www.desmos.com/calculator/ehu1yvja9z
# spreadsheet: https://docs.google.com/spreadsheets/d/1d8-onbCrB8yX4l5LnZqRzTZSysHMrqlJEVc6LUvenfE/edit?usp=sharing
# equation of best fit: y = (36.75131166 * (math.e ** (0.002864827 * cx))) + 18.65849

import math
import imutils
import numpy as np
import cv2
import glob

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_EXPOSURE,-13)

while True:

    ret, frame = cap.read()
    h, w, _ = frame.shape
    cam_x = int((w / 2) - 0.5)
    cam_y = int((h / 2) - 0.5)
    cam_center = (cam_x, cam_y)

    img = frame.copy()
    blur = cv2.blur(img, (4, 4))

    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(hsv, cv2.COLOR_BGR2GRAY)

    # mask = cv2.inRange(hsv, lower_bound, upper_bound)
    mask = cv2.inRange(hsv, (0, 200, 17), (255, 255, 255))
    result = cv2.bitwise_and(frame, frame, mask=mask)

    contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    if len(contours) != 0:
        c = max(contours, key=cv2.contourArea)
        M = cv2.moments(c)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])  # (testing)
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = 0, 0
        center = (cx, cy)
        print(cx)
        area = cv2.contourArea(c)
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 1)
        cv2.circle(frame, center, 1, (0, 255, 255), -1)
        distance = (36.75131166 * (math.e ** (0.002864827 * cx))) + 18.65849

        print(str(distance))

        cv2.imshow('Original', frame)
        # cv2.imshow('Mask', mask)
        key = cv2.waitKey(200)
        if key == 27:
            break

cap.release()
cv2.destroyAllWindows()
