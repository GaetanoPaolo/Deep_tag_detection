from matplotlib import pyplot as plt
import cv2 as cv
import os
import numpy as np
import h5py
import time
import crop
import itertools
import draw_transf as dw
import transform_mat
#load the logo template
logo_temp = cv.imread('/home/gaetan/code/simulation_ws/src/my_simulations/models/psi_logo/materials/textures/poster-psi-drone-logo-2percent.png',0)
logo_temp_color = cv.imread('/home/gaetan/code/simulation_ws/src/my_simulations/models/psi_logo/materials/textures/poster-psi-drone-logo.png')
#covering the inner logo to avoid keypoint confusion
rot = 0
plt.imshow(logo_temp),plt.show()
logo_temp = crop.crop_img(logo_temp)
# load the camera parameters stored in episode 1
f = h5py.File('/home/gaetan/data/hdf5/d400/data4_correct_gt.hdf5', 'r+')
base_items = list(f.items())
dset1 = f.get('1')
K = np.array(dset1.get('K')).reshape((3,3))
print(K)
# ILoad other parameters and images from chosen episode
dset2 = f.get('4')
group_items = list(dset2.items())
print(group_items)
imgs = np.array(dset2.get('observation'))
pos = np.array(dset2.get('position'))
quat = np.array(dset2.get('orientation'))
rel_pos = np.array(dset2.get('relative_position'))
img_stamp = np.array(dset2.get('image_time'))
pose_stamp = np.array(dset2.get('pose_time'))
observed_pos = 199
src = imgs[observed_pos,:,:,:]*255
src_gray = np.uint8(cv.cvtColor(src, cv.COLOR_BGR2GRAY))
plt.imshow(imgs[observed_pos,:,:,:]),plt.show()
T_gnd_odom = transform_mat.transf_mat(quat[observed_pos,:],pos[observed_pos,:])
T_odom_gnd = np.linalg.inv(T_gnd_odom)
pos_odom  = T_odom_gnd[0:3,3]
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
M, mask = cv.findHomography( pos_temp_hom, pos_target_hom,cv.RANSAC,2.0)
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
#print('Inlier match amount')
#print(len(inlier_matches))
ORB_inlier_matches =cv.drawMatches(logo_temp, kp_temp, src_gray, kp_target, inlier_matches, None, flags=2)
plt.imshow(ORB_inlier_matches),plt.show()

#apply solvepnp: https://learnopencv.com/head-pose-estimation-using-opencv-and-dlib/
pos_temp_world = dw.world_coord(np.array(pos_temp),logo_temp,rot)
dist_coeffs = np.zeros((4,1))
(suc,rot,trans) = cv.solvePnP(pos_temp_world, np.array(pos_target), K, dist_coeffs, flags=cv.SOLVEPNP_ITERATIVE)
#print(K)
#print(pos_temp_world)
#print(np.array(pos_target))
print('Estimated pos relative to cam')
print(trans)
print('cam pos relative to origin?')
print(pos[observed_pos,:])
print('origin pos relative to cam?')
print(pos_odom)
print(img_stamp[observed_pos,:]),
print(pose_stamp[observed_pos,:])
