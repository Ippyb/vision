# Find bounding box of the hub (ring)
# Make sure you know the width and height of the bounding box
# Find the coordinates for a) the middle of the right side of the rectangle and b) the middle of the left side
# Use both sets of coordinates to calculate 2 rays
# Find the angle between the two rays
# Use half of the angle, the radius of the hub, and do trig to get distance to the center of the hub


import math
import imutils
import config
import numpy as np
import cv2
import glob

data = []
distances = []

theta = math.pi/8
r = config.tall_hub_radius
r=26.75


# 3d points in real world space
objpoints = np.array([[[r*math.cos(0), r*math.sin(0), 0]
, [r*math.cos(theta), r*math.sin(theta), 0]
, [r * math.cos(2 * theta), r * math.sin(2 * theta), 0]
,[r * math.cos(3 * theta), r * math.sin(3 * theta), 0]
]], np.float32)



cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_EXPOSURE,-17)


def nothing(x):
    pass


cv2.namedWindow("HSV Adjustments")
cv2.namedWindow("Contour Adjustments")

cv2.createTrackbar("Lower_H", "HSV Adjustments", 0, 255, nothing)
cv2.createTrackbar("Lower_S", "HSV Adjustments", 0, 255, nothing)
cv2.createTrackbar("Lower_V", "HSV Adjustments", 0, 255, nothing)
cv2.createTrackbar("Upper_H", "HSV Adjustments", 0, 255, nothing)
cv2.createTrackbar("Upper_S", "HSV Adjustments", 0, 255, nothing)
cv2.createTrackbar("Upper_V", "HSV Adjustments", 0, 255, nothing)

cv2.createTrackbar("min_thresh", "Contour Adjustments", 0, 300, nothing)
cv2.createTrackbar("max_thresh", "Contour Adjustments", 0, 300, nothing)

theta = math.pi/8
r = config.tall_hub_radius


# 3d points in real world space
objpoints = np.array([[[r*math.cos(0), r*math.sin(0), 0]
, [r*math.cos(theta), r*math.sin(theta), 0]
, [r * math.cos(2 * theta), r * math.sin(2 * theta), 0]
,[r * math.cos(3 * theta), r * math.sin(3 * theta), 0]
]], np.float32)

while True:

    ret, frame = cap.read()
    h, w, _ = frame.shape
    cam_x = int((w / 2) - 0.5)
    cam_y = int((h / 2) - 0.5)
    cam_center = (cam_x, cam_y)

    img = frame.copy()
    blur = cv2.blur(img, (4, 4))

    # Custom HSV Filtering using trackbars

    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

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

    #mask = cv2.inRange(hsv, lower_bound, upper_bound)
    mask = cv2.inRange(hsv, (0,217,28), (92,255,215))
    result = cv2.bitwise_and(frame, frame, mask=mask)

    # find all lemon contours in mask
    contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    numContours = 0
    output = []

    # 2d points in real world space
    imagepoints = []

    # sort by x-coordinate of a countour (to get left-most to right-most)
    # when using the draw rectangle thing use the center coordinates of the right-most and left-most tape contours

    # function that will ultimately help return the x-coordinate of a contour later
    # def leftToRight(array):
    #     return array[1]
    #
    # def leftToRight2(array):
    #     return array[0]

    if len(contours) != 0:
        for c in contours:
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])  # (testing)
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = 0, 0
            center = [cx, cy]
            output.append([c, cx, cy, center])
            imagepoints.append(center)

        output.sort(key=lambda x: x[1])
        imagepoints.sort(key=lambda x: x[0])
        if len(imagepoints) > 4:
            imagepoints = imagepoints[len(imagepoints) - 4:len(imagepoints)]
        if len(output) > 4:
            output = output[len(output) - 4:len(output)]

        for o in output:
            x, y, w, h = cv2.boundingRect(o[0])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        imagepoints = np.array([imagepoints], np.float32)

        # Calibrate in real time
        #
        try:
            retval, rvecs, tvecs = cv2.solveP3P(objectPoints=objpoints, imagePoints=imagepoints,
                                            cameraMatrix=config.newcameramtx, distCoeffs=None, flags=cv2.SOLVEPNP_P3P)
        except:
            print("ERROR")
            continue
        # print(rvecs)
        # print("tvecs: " + str(tvecs))
        # rvec to rotation matrix by axisangle to 3 by 3
        rmatrix, _ = cv2.Rodrigues(np.array([rvecs[0][0][0], rvecs[0][1][0], rvecs[0][2][0]], np.float32))
        transposed = rmatrix.T
        tmatrix = np.array([tvecs[0][0][0], tvecs[0][1][0], tvecs[0][2][0]], np.float32).reshape(3, 1)
        real_cam_center = np.matmul(-transposed, tmatrix)

        print("real_cam_center: ", real_cam_center)
        distances.append(real_cam_center[1][0])


        # # find the bounding rectangles for the leftmost and rightmost reflective tape contours
        # leftmost_info = output[0]
        # leftmost_contour = leftmost_info[0]
        #
        # rightmost_info = output[-1]
        # rightmost_contour = rightmost_info[0]
        #
        # xl, yl, wl, hl = cv2.boundingRect(leftmost_contour)
        # xr, yr, wr, hr = cv2.boundingRect(rightmost_contour)
        #
        # # ---------------------------------------------------------------------------------------
        # # |                                                                                     |
        # # P1                                                                                   P2
        # # |                                                                                     |
        # # ---------------------------------------------------------------------------------------
        #
        # # find the rays from the camera center to P1 and P2, which are points on the hub outline
        # r1 = config.inverse_mat.dot([xl, yl + (hl/2), 1.0])
        # r2 = config.inverse_mat.dot([xr + wr, yr + (hr/2), 1.0])
        #
        # # Use dot product to find the angle between the two rays
        # cos_angle = (r1.dot(r2) / (np.linalg.norm(r1) * np.linalg.norm(r2)))
        # print("angle: " + str(cos_angle))
        # angle_radians = math.acos(cos_angle)
        # print("angle2: " + str(angle_radians))
        #
        # # Find the distance to the target. This calculation is dependent upon the assumption that the angles and
        # # distance are on a plane that is parallel to the top of the hub; we have not yet considered height
        # frontal_dist_to_target = config.tall_hub_radius / math.tan(cos_angle/2)
        #
        # # Using pythagorean thm to find the linear, straight-line distance to target
        # linear_dist_to_target = math.sqrt((frontal_dist_to_target * frontal_dist_to_target)
        #                                   + (config.tall_hub_height * config.tall_hub_height))
        #
        # #linear_dist_to_target = frontal_dist_to_target
        #
        # print("linear distance to target: " + str(linear_dist_to_target))
        #
        # # draw the 'human' left-most contour (in green)
        # cv2.rectangle(frame, (xl, yl), (xl + wl, yl + hl), (0, 255, 0), 2)
        # # draw the 'human' right-most contour (in green)
        # cv2.rectangle(frame, (xr, yr), (xr + wr, yr + hr), (0, 255, 0), 2)
        # # draw the 'human' whole hub contour (in green)
        # cv2.rectangle(frame, (xl, yl), (xr + wr, yr + hr), (0, 255, 0), 2)

        # draw the centers of the leftmost and rightmost contours
        cv2.circle(frame, cam_center, 2, (0, 255, 0), -1)
        # cv2.circle(frame, leftmost_info[3], 2, (0, 255, 0), -1)
        # cv2.circle(frame, rightmost_info[3], 2, (0, 255, 0), -1)
        # cv2.circle(frame, cam_center, 2, (0, 255, 0), -1)
        # cv2.circle(frame, (int(xl), int(yl + (hl/2))), 7, (0, 255, 0), -1)
        # cv2.circle(frame, (int(xr + wr), int(yr + (hr/2))), 7, (0, 255, 0), -1)

    cv2.imshow('Original', frame)
    cv2.imshow('Mask', mask)

    key = cv2.waitKey(100)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
