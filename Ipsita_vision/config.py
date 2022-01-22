import numpy as np


# [ [fx, 0, cx], [0, fy, cy], [0, 0, 1] ]
cameramtx = np.array([[662.3461673, 0, 293.51663222], [0, 664.22509824, 243.65759666],[  0, 0, 1]] )
dist = np.array([[2.47936730e-01, -2.34092515e+00, 2.73073943e-03, -1.58373364e-03, 7.08333922e+00]])

newcameramtx = np.array([[675.30944824, 0, 295.30676157], [0, 668.10913086, 243.85860464], [0, 0, 1]])

#mine
#newcameramtx= np.array([[1.12383716e+03, 0.00000000e+00, 5.33158579e+02], [0.00000000e+00, 1.06849597e+03, 4.85842118e+02],
#                        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])


roi = [] # TODO

focal_x = newcameramtx[0][0]
focal_y = newcameramtx[1][1]

cx = newcameramtx[0][2]
cy = newcameramtx[1][2]

cam_center = (cx, cy)

nt_ip = '10.1.92.2'

nt_name = 'jetson'

initial_camera_angle = 0

tape_width = 12.7

tape_height = 5.08

tall_hub_height = 83.82 # just using counter height + half of box right now; cm

short_hub_height = 0

