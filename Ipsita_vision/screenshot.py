# Find bounding box of the hub (ring)
# Make sure you know the width and height of the bounding box
# Find the coordinates for a) the middle of the right side of the rectangle and b) the middle of the left side
# Use both sets of coordinates to calculate 2 rays
# Find the angle between the two rays
# Use half of the angle, the radius of the hub, and do trig to get distance to the center of the hub


import math
import imutils
from Ipsita_vision.Final_files import config
import numpy as np
import cv2

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_EXPOSURE,-13)


def nothing(x):
    pass


i=0
distance = 264+44
height = 24.5 #when we put weirdly on chairs

while True:
    i = i + 1
    ret, frame = cap.read()

    if i%30 == 0:
        cv2.imwrite(str(distance) + "_" + str(i/30) + ".png" , frame)


    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
