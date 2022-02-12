# Evelyn's suggestions to test :/
# Go through all third-party functions and check that they return the same type/unit
# and also do the thing you think they do

# Measure angle from your test cases/images (take measurement to edge of hub as well and use trig)
# and get distance from there, should be the same, then use the computed angle from your test images
# and see if that is the same (if calculation is going wrong with angle or with distance)

# Go through each step and calculate by hand to see if you can yourself get an accurate distance on paper
# from pixel ratios

# Calibrate camera again and see if you get a different distortion matrix to see if that is ruining it
# or use Alina's matrix or a different camera, or a different computer though that reallyyyy shouldnt affect it



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

images = glob.glob('/Users/ipsita/PycharmProjects/GRT Vision 2021-2022/Ipsita_vision/109_images/*.png')

i = 0

for fname in images:
    i += 1
    frame = cv2.imread(fname)
    h, w, _ = frame.shape
    cam_x = int((w / 2) - 0.5)
    cam_y = int((h / 2) - 0.5)
    cam_center = (cam_x, cam_y)

    img = frame.copy()
    blur = cv2.blur(img, (4, 4))

    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(hsv, cv2.COLOR_BGR2GRAY)

    # mask = cv2.inRange(hsv, lower_bound, upper_bound)
    mask = cv2.inRange(hsv, (0, 210, 23), (255, 255, 255))
    result = cv2.bitwise_and(frame, frame, mask=mask)

    # find all lemon contours in mask
    contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    numContours = 0
    output = []


    # sort by x-coordinate of a countour (to get left-most to right-most)
    # when using the draw rectangle thing use the center coordinates of the right-most and left-most tape contours

    # function that will ultimately help return the x-coordinate of a contour later
    def leftToRight(array):
        return array[1]


    if len(contours) != 0:
        for c in contours:
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])  # (testing)
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = 0, 0
            center = (cx, cy)
            output.append([c, cx, cy, center])
        output.sort(key=leftToRight)

        # find the bounding rectangles for the leftmost and rightmost reflective tape contours
        leftmost_info = output[0]
        leftmost_contour = leftmost_info[0]

        rightmost_info = output[-1]
        rightmost_contour = rightmost_info[0]

        xl, yl, wl, hl = cv2.boundingRect(leftmost_contour)
        xr, yr, wr, hr = cv2.boundingRect(rightmost_contour)

        # ---------------------------------------------------------------------------------------
        # |                                                                                     |
        # P1                                                                                   P2
        # |                                                                                     |
        # ---------------------------------------------------------------------------------------

        # find the rays from the camera center to P1 and P2, which are points on the hub outline
        r1 = config.inverse_mat.dot([xl, yl + (hl / 2), 1.0])
        r2 = config.inverse_mat.dot([xr + wr, yr + (hr / 2), 1.0])

        # Use dot product to find the angle between the two rays
        cos_angle = (r1.dot(r2) / (np.linalg.norm(r1) * np.linalg.norm(r2)))
        print(i)
        print("angle: " + str(cos_angle))
        angle_radians = math.acos(cos_angle)
        print("angle2: " + str(angle_radians))

        # Find the distance to the target. This calculation is dependent upon the assumption that the angles and
        # distance are on a plane that is parallel to the top of the hub; we have not yet considered height
        #frontal_dist_to_target = config.tall_hub_radius / math.tan(cos_angle / 2)
        frontal_dist_to_target = config.tall_hub_radius / math.sin(cos_angle / 2)

        # Using pythagorean thm to find the linear, straight-line distance to target
        linear_dist_to_target = math.sqrt((frontal_dist_to_target * frontal_dist_to_target)
                                          + (config.tall_hub_height * config.tall_hub_height))

        # linear_dist_to_target = frontal_dist_to_target

        data.append(linear_dist_to_target)
        print("linear distance to target: " + str(linear_dist_to_target))

        # draw the 'human' left-most contour (in green)
        cv2.rectangle(frame, (xl, yl), (xl + wl, yl + hl), (0, 255, 0), 2)
        # draw the 'human' right-most contour (in green)
        cv2.rectangle(frame, (xr, yr), (xr + wr, yr + hr), (0, 255, 0), 2)
        # draw the 'human' whole hub contour (in green)
        cv2.rectangle(frame, (xl, yl), (xr + wr, yr + hr), (0, 255, 0), 2)

        # draw the centers of the leftmost and rightmost contours
        cv2.circle(frame, leftmost_info[3], 2, (0, 255, 0), -1)
        cv2.circle(frame, rightmost_info[3], 2, (0, 255, 0), -1)
        cv2.circle(frame, cam_center, 2, (0, 255, 0), -1)
        cv2.circle(frame, (int(xl), int(yl + (hl / 2))), 7, (0, 255, 0), -1)
        cv2.circle(frame, (int(xr + wr), int(yr + (hr / 2))), 7, (0, 255, 0), -1)
        cv2.imshow('Original', frame)
        cv2.imshow('Mask', mask)
        key = cv2.waitKey(1)
        if key == 27:
            break

print("109 images: " + str(data))
print("96 inches to center; mean: " + str(np.mean(config.new_seventy_images_filtered)) + ", SD: " + str((np.std(config.new_seventy_images_filtered))) + ", Variance: "
                                                                      +  str(np.var(config.new_seventy_images_filtered)))
print("116 inches to center; mean: " + str(np.mean(config.new_ninety_filtered)) + ", SD: " + str((np.std(config.new_ninety_filtered))) + ", Variance: "
                                                                      +  str(np.var(config.new_ninety_filtered)))
print("135 inches to center; mean: " + str(np.mean(config.new_onehundrednine_filtered)) + ", SD: " + str((np.std(config.new_onehundrednine_filtered))) + ", Variance: "
                                                                      +  str(np.var(config.new_onehundrednine_filtered)))
#