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
logo_temp = cv.imread('/home/gaetan/code/simulation_ws/src/my_simulations/models/psi_logo/materials/textures/poster-psi-drone-logo.png',0)
rot = 0

#removing the blue edge of the logo template to detect logo itself
#logo_temp = crop.crop_img(logo_temp,2)
plt.imshow(logo_temp),plt.show()
# load the labelled gazebo data
f = h5py.File('/home/gaetan/data/hdf5/psi_800res_hover/data4_sync.hdf5', 'r+')
base_items = list(f.items())
print(base_items)
dset = f.get('6')
imgs = np.array(dset.get('observation'))
corn = np.array(dset.get('corners'))
pos = np.array(dset.get('position'))
pos_origin_cam = np.array(dset.get('pos_origin_cam'))
print('image amount')
print(imgs.shape)
observed_pos = 30
src = imgs[observed_pos,:,:,:]*255
src_gray = np.uint8(cv.cvtColor(src, cv.COLOR_BGR2GRAY))
plt.imshow(src_gray),plt.show()

#code source till line 31: https://datahacker.rs/feature-matching-methods-comparison-in-opencv/
#creating keypoint matchers and finders
orb = cv.ORB_create(500,1.1,8,21,0,2,0,21,20)
bf = cv.BFMatcher_create(cv.NORM_HAMMING,crossCheck=True)

#creating mask where keypoints have to be found
cur_corn = corn[observed_pos,:,:]
corn_size = cur_corn.shape
img_size = src_gray.shape

#calculating keypoints and finding the matching ones
kp_temp, des_temp = orb.detectAndCompute(logo_temp,None)
print(len(kp_temp))
logo_temp_2 = cv.drawKeypoints(logo_temp,kp_temp,0,(255,0,0),cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
plt.imshow(logo_temp_2),plt.show()
kp_target, des_target = orb.detectAndCompute(src_gray,None)
print('des target')
print(len(des_target[0]))
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
# fx = 119
# fy = 119
horizontal_field_of_view = (80 * img_size[1]/img_size[0]) * 3.14 / 180
vertical_field_of_view = 80 * 3.14 / 180
fx = img_size[1]/2*np.tan(horizontal_field_of_view/2)**(-1)
fy = img_size[0]/2*np.tan(vertical_field_of_view/2)**(-1)
print(fx),print(fy)
K = np.array([[fx,0,img_size[1]/2],
                [0,fy,img_size[0]/2],
                [0,0,1]])
matched_temp_kpts = []
matched_target_kpts = []
for m in range(0,len(matches)):
    target_pt_idx = matches[m].trainIdx
    temp_pt_idx = matches[m].queryIdx
    temp_coord = kp_temp[temp_pt_idx].pt
    matched_temp_kpts.append(kp_temp[temp_pt_idx])
    target_coord = kp_target[target_pt_idx].pt
    matched_target_kpts.append(kp_target[target_pt_idx])
    pos_temp.append(temp_coord)
    pos_target.append(target_coord)

#apply solvepnp to estimate translation: https://learnopencv.com/head-pose-estimation-using-opencv-and-dlib/
pos_temp_world = dw.world_coord(np.array(pos_temp),logo_temp,rot)
dist_coeffs = np.zeros((4,1))
(suc,rot,trans,inliers) = cv.solvePnPRansac(pos_temp_world, np.array(pos_target), K, dist_coeffs, flags=cv.SOLVEPNP_ITERATIVE, iterationsCount=2000, reprojectionError=8.0)
# [H,mask] = cv.findHomography(pos_temp_world, np.array(pos_target),cv.RANSAC,2.0,2000)
# [retval,rot,trans,normals] = cv.decomposeHomographyMat(H, K)
print('Inlier match amount')
print(len(inliers))
inlier_matches = []
for k in range(0,len(inliers)):
    inlier_matches.append(matches[inliers[k][0]])
ORB_matches =cv.drawMatches(logo_temp, kp_temp, src_gray, kp_target, inlier_matches, None, flags=2)
plt.imshow(ORB_matches),plt.show() 
print('Estimate origin pose in camera coord')
print(trans)


#computing the error between estimated relative pose and GT
print('GT in camera coordinates')
print(pos_origin_cam[observed_pos,:])
diff = np.subtract(pos_origin_cam[observed_pos,0:3],trans)
print('Diff')
print(diff)
#print("GT range")
#print(pos_origin_cam[observed_pos-10:observed_pos+10,0:3])



        


