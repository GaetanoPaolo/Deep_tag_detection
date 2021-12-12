from matplotlib import pyplot as plt
import cv2 as cv
import os
import numpy as np
import h5py
import time
import crop
import itertools
import draw_transf as dw
#load the logo template
logo_temp = cv.imread('/home/gaetan/code/simulation_ws/src/my_simulations/models/psi_logo/materials/textures/poster-psi-drone-logo-2percent.png',0)
logo_temp_color = cv.imread('/home/gaetan/code/simulation_ws/src/my_simulations/models/psi_logo/materials/textures/poster-psi-drone-logo.png')
#covering the inner logo to avoid keypoint confusion
rot = 0
plt.imshow(logo_temp),plt.show()
#removing the blue edge of the logo template to detect logo itself
logo_temp = crop.crop_img(logo_temp)
# load the labelled gazebo data
f = h5py.File('/home/gaetan/data/hdf5/correct_baselink_gt/data4_correct_gt.hdf5', 'r+')
base_items = list(f.items())
dset = f.get('labelled_data')
imgs = np.array(dset.get('observation'))
corn = np.array(dset.get('corners'))
pos = np.array(dset.get('position'))
pos_origin_cam = np.array(dset.get('pos_origin_cam'))
observed_pos = 849
src = imgs[observed_pos,:,:,:]*255
src_gray = np.uint8(cv.cvtColor(src, cv.COLOR_BGR2GRAY))

#code source till line 31: https://datahacker.rs/feature-matching-methods-comparison-in-opencv/
#creating keypoint matchers and finders
orb = cv.ORB_create()
bf = cv.BFMatcher_create(cv.NORM_HAMMING,crossCheck=True)
#calculating keypoints and finding the matching ones

kp_temp, des_temp = orb.detectAndCompute(logo_temp,None)
#print(kp_temp[1].pt)
kp_target, des_target = orb.detectAndCompute(src_gray,None)
matches = bf.match(des_temp, des_target)
#import pdb; pdb.set_trace()
matches = sorted(matches,key=lambda x:x.distance)
print('Match amount')
print(len(matches))
ORB_matches =cv.drawMatches(logo_temp, kp_temp, src_gray, kp_target, matches[0:len(matches)], None, flags=2)
plt.imshow(ORB_matches),plt.show()

pos_temp = []
pos_target = []
#2) approximating camera parameters, will be approximated again during solvepnp
# if result incorrect => calibrate camera_
fx = 119
fy = 119
img_size = src_gray.shape
K = np.array([[fx,0,img_size[1]/2],
                [0,fy,img_size[0]/2],
                [0,0,1]])

for m in range(0,len(matches)):
    target_pt_idx = matches[m].trainIdx
    temp_pt_idx = matches[m].queryIdx
    temp_coord = kp_temp[temp_pt_idx].pt
    target_coord = kp_target[target_pt_idx].pt
    pos_temp.append(temp_coord)
    pos_target.append(target_coord)
pos_temp_hom = np.float32(pos_temp).reshape(-1,1,2)
pos_target_hom = np.float32(pos_target).reshape(-1,1,2)
#Homography takes point inputs as numpy vectors with lists of points as elements
M, mask = cv.findHomography( pos_temp_hom, pos_target_hom,cv.RANSAC,5.0)
matchesMask = mask.ravel().tolist()
inlier_matches = []
rem = 0
for k in range(0,len(matches)):
    if matchesMask[k] == 1:
        inlier_matches.append(matches[k])
    else:
        pos_temp.pop(k-rem)
        pos_target.pop(k-rem)
        rem += 1
print('Inlier match amount')
print(len(inlier_matches))
ORB_inlier_matches =cv.drawMatches(logo_temp, kp_temp, src_gray, kp_target, inlier_matches, None, flags=2)
plt.imshow(ORB_inlier_matches),plt.show()

#apply solvepnp: https://learnopencv.com/head-pose-estimation-using-opencv-and-dlib/
pos_temp_world = dw.world_coord(np.array(pos_temp),logo_temp,rot)
dist_coeffs = np.zeros((4,1))
(suc,rot,trans) = cv.solvePnP(pos_temp_world, np.array(pos_target), K, dist_coeffs, flags=cv.SOLVEPNP_ITERATIVE)
print(K)
print(pos_temp_world)
print(np.array(pos_target))
print(trans)
#computing the translation in world coordinates 
cur_corn = corn[observed_pos,:,:]
corn_size = cur_corn.shape
corn_tup = []
for i in range(0,corn_size[1]):
    corn_tup.append((float(cur_corn[0,i]),float(cur_corn[1,i])))
    
abs_y_max = 0.68/2
abs_x_max = 0.98/2
z = 0.001

c_world = [(abs_x_max,-abs_y_max,z),(-abs_x_max,-abs_y_max,z),(-abs_x_max,abs_y_max,z),(abs_x_max,abs_y_max,z)]
# imgpts, jac = cv.projectPoints(np.float32(c_world), rot, pos_cam_world[observed_pos,:], np.float32(K), dist_coeffs)
# img1 = imgs[observed_pos,:,:,:]
# for point in imgpts:
#     pt = point[0,:]
#     cv.circle(img1, (int(pt[1]),int(pt[0])), radius=2, color=(255, 0, 0), thickness=1)
# plt.imshow(img1),plt.show()
(suc_corn,rot_corn,trans_corn) = cv.solvePnP(np.array(c_world), np.array(corn_tup), K, dist_coeffs, flags=cv.SOLVEPNP_ITERATIVE)
print(trans_corn)
print(pos_origin_cam[observed_pos,:])
print(pos[observed_pos,:])
img2 = imgs[observed_pos,:,:,:]
keypts = []
for j in range(0,corn_size[1]):
    keypts.append((int(cur_corn[0,j]),int(cur_corn[1,j])))
cv.drawContours(img2,[np.array(keypts)],0,(255,0,0),1)
plt.imshow(img2),plt.show()  


        


