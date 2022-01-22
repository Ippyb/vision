import math

import numpy as np
import cv2
import glob

import config
# save numpy array as csv file
from numpy import asarray
from numpy import savetxt

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((7*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:7].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('/Users/ipsita/PycharmProjects/GRT Vision 2021-2022/Ipsita_vision/Chessboardpics/*.jpg')

count = 1;
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #print("in image", count)
    count += 1

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (9,7),None)
    #print("ret", ret)

    # If found, add object points, image points (after refining them)
    if ret == True:
        #print("in ret=true")
        objpoints.append(objp)
        #print("got appended to obj")

        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (9,7), corners2,ret)
        cv2.imshow('img',img)
        cv2.waitKey(500)

cv2.destroyAllWindows()
#print("imagepoints length is", len(imgpoints))
#print("objectpoints length is", len(objpoints))

#Calibrate camera
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

img = cv2.imread('actualTest.jpg')
h,  w = img.shape[:2]
newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
# undistort
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

# crop the image
x,y,w,h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite('calibresult.png',dst)
# undistort
mapx,mapy = cv2.initUndistortRectifyMap(mtx,dist,None,newcameramtx,(w,h),5)
dst = cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)

# crop the image
x,y,w,h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite('calibresult.png',dst)

# do i use newcameramatrix or mtx?
#[rvec2, tvec2, success] = cv2.solvePnP(objpoints, imgpoints, newcameramtx, dist)

distance = math. sqrt(tvecs[0][0]**2 + tvecs[2][0]**2)

print(ret)
print()
print(newcameramtx)
print()
print(dist)
print()
print(rvecs)
print()
print(tvecs)
print()
print(distance)

def get_cube_values_calib(self, center):


#Calculate the angle and distance from the camera to
#the center point of the ball. This routine uses the cameraMatrix
#from the calibration to convert to normalized coordinates



# use the distortion and camera arrays to correct
# the location of the center point
# got this from
# https://stackoverflow.com/questions/8499984/
# how-to-undistort-points-in-camera-shot-coordinates-and-obtain-corresponding-undi


    center_np = np.array([[[float(self.center[0]), float(self.center[1])]]])
    out_pt = cv2.undistortPoints(center_np, self.cameraMatrix, self.distortionMatrix,
    P=self.cameraMatrix)
    undist_center = out_pt[0, 0]
    x_prime = (undist_center[0] - self.cameraMatrix[0, 2]) / self.cameraMatrix[0, 0]
    y_prime = -(undist_center[1] - self.cameraMatrix[1, 2]) / self.cameraMatrix[1, 1]
    # now have all pieces to convert to horizontal angle:
    ax = math.atan2(x_prime, 1.0)
    # corrected expression.
    # As horizontal angle gets larger, real vertical angle gets a little smaller
    ay = math.atan2(y_prime * math.cos(ax), 1.0)
    # now use the x and y angles to calculate the distance to the target:
    d = (self.target_height - self.camera_height) / math.tan(self.tilt_angle + ay)
    print("ax: " + str(ax) + ", ay: " + str(ay) + ", distance: " + str(d))
    #return ax, d # return horizontal angle and distance
