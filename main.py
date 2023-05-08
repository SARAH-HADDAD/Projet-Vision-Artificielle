import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Calibrez votre caméra:

focal_fx = np.load('./data/mtx.npy')[0,0]
distance_b = 6.1

def show_image(img, name='Image'):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Acquérir deux images en effectuant un mouvement de translation sur un plan:
imgL = cv2.imread('./pics/im1.jpg')
imgR = cv2.imread('./pics/im2.jpg')

imgL = cv2.resize(imgL,(1000, 800))
imgR = cv2.resize(imgR,(1000, 800))

imgL = cv2.cvtColor(imgL,cv2.COLOR_BGR2GRAY)
imgR = cv2.cvtColor(imgR,cv2.COLOR_BGR2GRAY)

show_image(imgL, 'ImageLeft')
show_image(imgR, 'ImageRight')

# Localisez les points SIFT et réalisez leur mise en correspondance:

sift = cv2.SIFT_create()

kpL, desL = sift.detectAndCompute(imgL,None)
kpR, desR = sift.detectAndCompute(imgR,None)

sift_imgL=cv2.drawKeypoints(imgL,kpL,imgL,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
sift_imgR=cv2.drawKeypoints(imgR,kpR,imgR,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

show_image(sift_imgL, 'SIFT_ImageLeft')
show_image(sift_imgR, 'SIFT_ImageRight')

match = cv2.BFMatcher()
matches = match.knnMatch(desL,desR,k=2)

threshold = 15
good = []

ul_list = []
ur_list = []

for m,n in matches:
    if m.distance < 0.75*n.distance:
        kp1 = kpL[m.queryIdx]
        kp2 = kpR[n.trainIdx]
        if abs(kp1.pt[1] - kp2.pt[1]) < threshold: 
            good.append([m])
            ul_list.append(kp1.pt[0])
            ur_list.append(kp2.pt[0])


query_pts = np.float32([kpL[m[0].queryIdx].pt for m in good ]).reshape(-1, 1, 2)
train_pts = np.float32([kpR[m[0].trainIdx].pt for m in good ]).reshape(-1, 1, 2)

H, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 3, None, 100, 0.99)
matches = mask.ravel().tolist()

ransac_good = []
i=0
for m in good:
    if matches[i]:
        ransac_good.append(m)
    i=i+1


good = ransac_good

img_res = cv2.drawMatchesKnn(imgL,kpL,imgR,kpR,good, None, flags=2)

show_image(img_res, 'Matches')

#Calculez les coordonnées 3D des points associés aux paires de points SIFT appariés:

X = []
Y = []
Z = []

for i,g in enumerate(good) : 
    x,y = kpL[g[0].queryIdx].pt
    z = (distance_b * focal_fx) / abs((ul_list[i]-ur_list[i]))
    X.append(int(x))
    Y.append(int(y))
    Z.append(int(z))

X = np.array(X)
Y = np.array(Y)
Z = np.array(Z)

# Affichez les points 3D dans un nuage de points 3D:

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')

# Customize the appearance of the scatter plot
ax.scatter(X, Y, Z, s=30, c='blue', marker='o', alpha=0.5)

# Set the labels for the axes
ax.set_xlabel('X', fontsize=12, labelpad=10)
ax.set_ylabel('Y', fontsize=12, labelpad=10)
ax.set_zlabel('Z', fontsize=12, labelpad=10)

# Set the limits for each axis
ax.set_xlim([min(X), max(X)])
ax.set_ylim([min(Y), max(Y)])
ax.set_zlim([min(Z), max(Z)])

# Set the title for the plot
ax.set_title('3D Scatter Plot', fontsize=16, pad=20)

plt.show()



