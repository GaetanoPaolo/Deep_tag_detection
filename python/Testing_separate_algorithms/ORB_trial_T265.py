from matplotlib import pyplot as plt
import cv2 as cv
import os
import numpy as np
import h5py
import time
import crop
import itertools
import draw_transf as dw
import transform_mat as tm
#load the logo template
logo_temp = cv.imread('/home/gaetan/code/simulation_ws/src/my_simulations/models/psi_logo/materials/textures/poster-psi-drone-logo-1percent.png',0)
logo_temp_color = cv.imread('/home/gaetan/code/simulation_ws/src/my_simulations/models/psi_logo/materials/textures/poster-psi-drone-logo.png')
#The scale var indicates the percentage kept from the original logo resolution
scale = 1
rot = 0
plt.imshow(logo_temp),plt.show()
logo_temp = crop.crop_img(logo_temp,scale)
# load the camera parameters stored in episode 1
f = h5py.File('/home/gaetan/data/hdf5/T265/data4_sync.hdf5', 'r+')
base_items = list(f.items())
dset1 = f.get('1')
K = np.array(dset1.get('K')).reshape((3,3))
# ILoad other parameters and images from chosen episode
dset2 = f.get('1')
print(base_items)
group_items = list(dset2.items())
imgs = np.array(dset2.get('observation'))
pos = np.array(dset2.get('position'))
quat = np.array(dset2.get('orientation'))
rel_pos = np.array(dset2.get('relative_position'))
img_stamp = np.array(dset2.get('image_time'))
pose_stamp = np.array(dset2.get('pose_time'))
observed_pos =265
src = imgs[observed_pos,:,:,:]*255
src_gray = np.uint8(src)
plt.imshow(imgs[observed_pos,:,:,:]),plt.show()
print(len(imgs))

#code source till line 31: https://datahacker.rs/feature-matching-methods-comparison-in-opencv/
#creating keypoint matchers and finders
#code source till line 31: https://datahacker.rs/feature-matching-methods-comparison-in-opencv/
#creating keypoint matchers and finders
#orb = cv.ORB_create(2000,1.1,8,11,0,2,0,11,10)
orb = cv.ORB_create(2000,1.1,8,11,0,2,0,11,5)
bf = cv.BFMatcher_create(cv.NORM_HAMMING,crossCheck=True)

#avoid detecting the camera lens edge among the keypoints
src_gray = src_gray[144:657,144:705]
#ret,src_gray = cv.threshold(src_gray,100,255,cv.THRESH_BINARY)
#ret2,logo_temp = cv.threshold(logo_temp,200,255,cv.THRESH_BINARY)
#Gaussian denoising of the grayscale image
#src_gray = cv.GaussianBlur(src_gray,(5,5),0,0,borderType = cv.BORDER_CONSTANT)

#calculating keypoints and finding the matching ones
kp_temp, des_temp = orb.detectAndCompute(logo_temp,None)
print(len(kp_temp))
logo_temp_2 = cv.drawKeypoints(logo_temp,kp_temp,0,(255,0,0),cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
plt.imshow(logo_temp_2),plt.show()
kp_target, des_target = orb.detectAndCompute(src_gray,None)
print(len(kp_target))
src_gray_2 = cv.drawKeypoints(src_gray,kp_target,0,(255,0,0),cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
plt.imshow(src_gray_2),plt.show()
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
matched_temp_kpts = []
matched_target_kpts = []
for m in range(0,len(matches)):
    target_pt_idx = matches[m].trainIdx
    temp_pt_idx = matches[m].queryIdx
    temp_coord = kp_temp[temp_pt_idx].pt
    matched_temp_kpts.append(kp_temp[temp_pt_idx])
    target_coord = kp_target[target_pt_idx].pt
    target_coord = (target_coord[0]+144,target_coord[1]+144)
    matched_target_kpts.append(kp_target[target_pt_idx])
    pos_temp.append(temp_coord)
    pos_target.append(target_coord)
#apply solvepnp to estimate translation: https://learnopencv.com/head-pose-estimation-using-opencv-and-dlib/
pos_temp_world = dw.world_coord(np.array(pos_temp),logo_temp,rot)
dist_coeffs = np.zeros((4,1))
(suc,rot,trans,inliers) = cv.solvePnPRansac(pos_temp_world, np.array(pos_target), K, dist_coeffs, flags=cv.SOLVEPNP_ITERATIVE, iterationsCount=2000, reprojectionError=8)
print('Inlier match amount')
print(len(inliers))
inlier_matches = []
for k in range(0,len(inliers)):
    inlier_matches.append(matches[inliers[k][0]])
ORB_matches =cv.drawMatches(logo_temp, kp_temp, src_gray, kp_target, inlier_matches, None, flags=2)
plt.imshow(ORB_matches),plt.show() 

#converting the relative position to camera axes
#the given rel_pos is from the tag to the drone in world coordinates
#Converting to position from drone to tag:
rel_pos_w = -rel_pos[observed_pos,:]
#First, let us consider a rotation matrix of the axes attched to the drone relative to 
#the world axes. This one is obtained from the given roation quaterion.
#Computing the roation matrix from the world axes to the axes fixed to the drone
T_w_d = tm.transf_mat(quat[observed_pos,:],np.zeros((3,1)))
R_w_d = T_w_d[0:3,0:3]
#Secondly, the axes of the camera do not correspond to the ones attached to the drone. 
#Taking the fixed axis on the drone identical to the world axes (NWU), with the front of the drone poining south. 
#The transition from drone to camera happens with the following matrix:
R_d_c = np.matrix([[0,1,0],
                [1,0,0],
                [0,0,-1]])
#Computing
rel_pos_d = np.matmul(R_w_d,np.array(rel_pos_w))
rel_pos_c = np.matmul(R_d_c,np.transpose(rel_pos_d))
print('Estimated pos relative to cam')
print(trans)
print('origin pos relative to cam')
print(rel_pos_c)
