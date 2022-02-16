import numpy as np


# [ [fx, 0, cx], [0, fy, cy], [0, 0, 1] ]
cameramtx = np.array([[662.3461673, 0, 293.51663222], [0, 664.22509824, 243.65759666],[  0, 0, 1]] )
dist = np.array([[2.47936730e-01, -2.34092515e+00, 2.73073943e-03, -1.58373364e-03, 7.08333922e+00]])

#newcameramtx = np.array([[675.30944824, 0, 295.30676157], [0, 668.10913086, 243.85860464], [0, 0, 1]])



#mine
newcameramtx= np.array([[1.12383716e+03, 0.00000000e+00, 5.33158579e+02], [0.00000000e+00, 1.06849597e+03, 4.85842118e+02],
                       [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
inverse_mat = np.linalg.inv(newcameramtx)

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

tall_hub_radius = 53.13/2

tall_hub_height = 1 # just using counter height + half of box right now; cm

short_hub_height = 0

sensor_width = 44.5 #mm

# seventy_images = [99.24082121827965, 100.3499759997589, 100.3499759997589, 99.37508430257583, 99.40543018082542,
#                   100.35843057444333, 100.11628926387563, 100.2730777787591, 100.2184896592145, 100.30367286646425,
#                   100.05823046055455, 99.24158316587128, 99.08103332248004, 99.02776671877295, 99.87220473123719,
#                   99.96041108731994, 98.82204939671821, 98.83924659644109, 98.85221767262107, 98.85471178668469,
#                   98.82566546606738, 99.74129779527448, 99.24158316587128, 99.7423805218683, 99.59345302494192,
#                   99.5546589178775, 98.55868615357772, 98.63571187186442, 98.62835697140936, 98.63673282690434,
#                   98.85258441695052, 99.71229797740554, 99.54946632073562, 99.26880628922818, 99.550454220567,
#                   99.63159249201887, 98.7204569360573, 98.76848437036166, 98.75438009523852, 99.265668618861,
#                   100.33232940260349, 100.398852236477, 100.48178133081248, 100.50195411032435]
# seventy_images_filtered = [100.30367286646425, 99.96041108731994, 99.74129779527448, 99.54946632073562,
#                            99.550454220567]

new_seventy = [57.9573361336529, 57.720970809595116, 57.50272188346641, 57.721832803748285, 57.64338726069487, 57.97835517077641, 57.93075294436175, 57.55022944103735, 57.257051523194754, 57.49540421840836, 57.49485483776628, 57.25825927653547, 57.95644540370641, 57.95103701904173, 57.226509902261235, 57.91186497737623, 57.33044610275342, 57.85847756485892, 57.65891055292547, 57.506454915133375, 57.25385558239236, 57.95644540370641, 57.559273123428376, 57.49540421840836, 57.34680573094154, 57.315467637469624, 57.77152360077975, 57.43807814377698, 57.679774576322615, 57.64390062805564, 58.01401945681134, 57.84629599272762, 57.33391117306918, 57.3439104901532, 57.41296682858977, 57.71184371928681, 57.29708584814456, 57.808546281667965, 57.34475424128544, 57.309040561646036, 58.0196811567778, 57.33945190523233, 57.94809949369788]


new_seventy_images_filtered = [new_seventy[9], new_seventy[15], new_seventy[21], new_seventy[32], new_seventy[34]]

# ninety_images = [98.54082335203766, 98.49937152653513, 98.57434949521007, 98.70369424586113, 98.35800595208102,
#                  97.72438345601725, 98.29418970456567, 98.34478972646887, 98.42254141447388, 98.39373636407225,
#                  97.66241223859792, 98.54082335203766, 97.7351181674365, 97.78019983786932, 97.81553967089985,
#                  98.5217987721976, 98.64552189250178, 98.80335564990301, 98.80733438660093, 98.60056039657786,
#                  98.54082335203766, 98.56611073430382, 98.65106962859937, 98.43263436216002, 97.80407689802459,
#                  97.79323166644126, 97.79971139689988]
# ninety_images_filtered = [98.49937152653513, 98.54082335203766, 98.80335564990301, 98.54082335203766]

new_ninety = [57.1923406063942, 56.8833233270604, 57.13943106469795, 57.30846903405452, 56.8896534293587, 57.224394225342856, 56.86522193083324, 56.89684290085001, 56.88762399112, 57.31050772362001, 57.24388943051666, 56.835843127146525, 57.154196234669385, 57.097650202980276, 57.15069495552328, 57.21104635165523, 57.12355317669049, 57.214145281537796, 57.20057365236994, 57.118648268857186, 57.2453740517608, 56.893244425906424, 56.85991921667074, 57.183295709037125, 57.20057365236994, 57.20057365236994, 57.267287942804074]


new_ninety_filtered = [new_ninety[0], new_ninety[11], new_ninety[17], new_ninety[20]]


# onehundrednine_images = [96.9261678337379, 97.65768837592873, 97.51321854064409, 97.53400776848348, 95.2110445899022,
#                          96.98492236459872, 96.98949698340037, 97.4953467190356, 97.5088932381784, 97.48902201747553,
#                          97.46925369554576, 96.9261678337379, 97.41382125647503, 96.9419653314052, 97.41515919599111,
#                          97.5023190886015, 97.50450036755484, 97.41573595775807, 97.4972972261227, 97.49680310688191,
#                          97.53604382390294, 97.55553045205612, 97.64726767342584, 97.64852881026249]
# onehundrednine_images_filtered = [97.65768837592873, 97.5088932381784, 97.5023190886015, 97.53604382390294,
#                                   97.64852881026249]

new_onehundrednine = [56.55437802314049, 56.772612017231026, 56.8308465210096, 55.813320703595096, 56.774077162835276, 56.76671008641373, 56.55313748533958, 56.834237465901936, 56.831071228904634, 56.78270195119381, 56.77033860548828, 56.76899910469479, 56.78533949472132, 56.535836496599515, 56.76409071001057, 56.529557686760725, 56.79355902382648, 56.73202678705923, 56.73413031993631, 56.7686306000014, 56.529557686760725, 56.75561642210724, 56.77097993330735, 56.736021681836334]

new_onehundrednine_filtered = [new_onehundrednine[1], new_onehundrednine[8], new_onehundrednine[15], new_onehundrednine[20], new_onehundrednine[23]]

##Values from testing as of 2/12/22
# In actuality its 96 inches to center; calculated values are mean: 61.55472297327715, SD: 0.7740986070821999, Variance: 0.599228653486602
# In actuality its 116 inches to center; calculated values are mean: 59.77048838188013, SD: 0.4794457828577414, Variance: 0.22986825870007252
# In actuality its 135 inches to center; calculated values are mean: 58.49896752892566, SD: 0.3966782497233383, Variance: 0.15735363380357115