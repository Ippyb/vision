# Using known shape size (width of tape) and scaling (plus some trig probably) to find distance
# center the reflective tape in the camera fov
# measure the vertical angle from cam to tape

# We have seen that the retro-reflective tape will not shine unless a light source
# is directed at it, and the light source must pass very near the camera lens or the observer’s eyes
# Thus, we use a ring light bc they work quite well for causing retro-reflective tape to shine

import math
import imutils
import config
import numpy as np
import cv2
import glob

# pitch_angle = math.degrees(math.atan((y_cam_coord - 243.65759666) / config.focal_len))
# yaw_angle = math.degrees(math.atan((x_cam_coord - 293.51663222) / config.focal_len))

# Next step is to use thresholding. Thresholding is taking an image, and throwing away any pixels
# that aren’t in a specific color range. The result of thresholding is generally a one-dimensional
# image in which a pixel is either “on” or “off. We'll use HSV to specify the color of the target
##gray = cv2.cvtColor(cFrame, cv2.COLOR_BGR2GRAY)
##blurred = cv2.blur(gray, (4, 4))

import cv2
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_EXPOSURE,-13)


def nothing(x):
    pass


cv2.namedWindow("HSV Adjustments")
cv2.namedWindow("Contour Adjustment")

cv2.createTrackbar("Lower_H", "HSV Adjustments", 0, 255, nothing)
cv2.createTrackbar("Lower_S", "HSV Adjustments", 0, 255, nothing)
cv2.createTrackbar("Lower_V", "HSV Adjustments", 0, 255, nothing)
cv2.createTrackbar("Upper_H", "HSV Adjustments", 0, 255, nothing)
cv2.createTrackbar("Upper_S", "HSV Adjustments", 0, 255, nothing)
cv2.createTrackbar("Upper_V", "HSV Adjustments", 0, 255, nothing)

cv2.createTrackbar("min_thresh", "Contour Adjustments", 0, 300, nothing)
cv2.createTrackbar("max_thresh", "Contour Adjustments", 0, 300, nothing)

while True:

    ret, frame = cap.read()
    h, w, _ = frame.shape
    cam_x = int((w / 2) - 0.5)
    cam_y = int((h / 2) - 0.5)
    cam_center = (cam_x, cam_y)
    cv2.circle(frame, cam_center, 7, (0, 255, 0), -1)

    # as per https://github-wiki-see.page/m/AIRLegend/aitrack/wiki/Common-camera-FOV-values
    # https://github.com/Wibis428/FRC1787-Hoff10k/blob/master/FRC1787-Hoff10k/src/org/usfirst/frc/team1787/robot/vision/CameraController.java
    #fovy = 68.5
    #fovy = 36.93
    #degrees_per_pixel = fovy / h
    diagonal_fov = 68.5
    #according to http://kmp.pentaxians.eu/technology/fov/#:~:text=The%20formula%20that%20it%20implements,smaller%20across%20the%20vertical%20dimension.
    foc_len = 21.1818452699873


    img = frame.copy()
    blur = cv2.blur(img, (4, 4))

    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(hsv, cv2.COLOR_BGR2GRAY)

    l_h = cv2.getTrackbarPos("Lower_H", "HSV Adjustments")
    l_s = cv2.getTrackbarPos("Lower_S", "HSV Adjustments")
    l_v = cv2.getTrackbarPos("Lower_V", "HSV Adjustments")
    u_h = cv2.getTrackbarPos("Upper_H", "HSV Adjustments")
    u_s = cv2.getTrackbarPos("Upper_S", "HSV Adjustments")
    u_v = cv2.getTrackbarPos("Upper_V", "HSV Adjustments")

    min_thresh = cv2.cv2.getTrackbarPos("min_thresh", "Contour Adjustments")
    max_thresh = cv2.cv2.getTrackbarPos("max_thresh", "Contour Adjustments")

    lower_bound = np.array([l_h, l_s, l_v])
    upper_bound = np.array([u_h, u_s, u_v])

    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    #mask = cv2.inRange(hsv, (76,64,243), (121,161,255))
    result = cv2.bitwise_and(frame, frame, mask=mask)

    # find all lemon contours in mask
    contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    numContours = 0
    output = []

    # sort by area
    # when using the draw rectangle thing use the center coordinates of the largest area

    if len(contours) != 0:
        # the contours are drawn here
        # cv2.drawContours(frame, contours, -1, 255, 3)

        # find the biggest area of the contour
        c = max(contours, key=cv2.contourArea)

        M = cv2.moments(c)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])  # (testing)
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = 0, 0
        center = (cx, cy)

        yaw_angle = math.degrees(math.atan((cx - cam_x) / config.focal_x))
        '''
        if (cy - cam_y != 0):
            pitch_angle = -math.degrees(math.atan((cy - cam_y) / config.focal_y))
            x_dist_to_target = abs(config.tall_hub_height / (math.tan(pitch_angle)))
        else:
            x_dist_to_target = 0

        vert_dist_to_target = abs(config.tall_hub_height * math.sin(pitch_angle))
        y_dist_to_target = math.sqrt(
            (abs((vert_dist_to_target * vert_dist_to_target) - (x_dist_to_target * x_dist_to_target))))

        print("pitch: " + str(pitch_angle) + ", yaw: " + str(yaw_angle))
        print("x_dist: " + str(x_dist_to_target) + ", y-dist: " + str(y_dist_to_target))
        '''
        x, y, w, h = cv2.boundingRect(c)
        # draw the 'human' contour (in green)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        # cv2.drawContours(frame, contours, 0, (0, 255, 0), 3)
        cv2.circle(frame, center, 4, (0, 255, 0), -1)

        up_dist_to_target = config.tall_hub_height
        if (cy - cam_y != 0):
            pitch_angle_d = -math.degrees(math.atan((cy - cam_y) / config.focal_y))
            pitch_angle_r = math.atan((cy - cam_y) / config.focal_y)
            ##########
            ##pitch_angle_d = abs(cy - cam_y)*degrees_per_pixel
            ##pitch_angle_r = pitch_angle_d*(math.pi/180)
            ##########
            # pitch_angle_r = 24*(math.pi/180)
            #pitch_angle_d = -math.degrees(math.atan((cy - cam_y) / foc_len))
            #pitch_angle_r = math.atan((cy - cam_y) / foc_len)
            front_dist_to_target = abs(up_dist_to_target / math.tan(pitch_angle_r))
            # front_dist_to_target = abs(up_dist_to_target / math.tan(30*(math.pi/180)))
        else:
            front_dist_to_target = 0

        linear_dist_to_target = math.sqrt(
            (abs((up_dist_to_target * up_dist_to_target) + (front_dist_to_target * front_dist_to_target))))

        print("pitch: " + str(pitch_angle_d) + ", yaw: " + str(yaw_angle))
        print("front_dist: " + str(front_dist_to_target) + ", linear_dist: " + str(linear_dist_to_target))

        # print(config.cam_center)
    '''



            pitch_angle = math.degrees(math.atan((cy - 243.65759666) / config.focal_len))
            yaw_angle = math.degrees(math.atan((cx - 293.51663222) / config.focal_len))

            #qprint("area is: " + str(area))
            #print("centroid is at: " + str(cx) + "," + str(cy))
            #print("pitch: " + str(pitch_angle) + ", yaw: " + str(yaw_angle))
    '''

    cv2.imshow('Original', frame)
    cv2.imshow('Mask', mask)

    key = cv2.waitKey(1)
    if key == 27:
        print("l_h:" + str(l_h) + " " + "l_s:" + str(l_s) + " " + "l_v:" + str(l_v) + " " +
              "u_h: " + str(u_h) + " " + "u_s:" + str(u_s) + " " + "u_v:" + " " + str(u_v))
        break

cap.release()
cv2.destroyAllWindows()
