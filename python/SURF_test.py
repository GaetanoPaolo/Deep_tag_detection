from matplotlib import pyplot as plt

import cv2 as cv
import os
import numpy as np
import h5py
import time
import crop
#load the logo template
logo_temp = cv.imread('/home/gaetan/code/simulation_ws/src/my_simulations/models/psi_logo/materials/textures/poster-psi-drone-logo.png',0)
logo_temp = crop.crop_img(logo_temp)
# load the labelled gazebo data
f = h5py.File('/home/gaetan/data/hdf5/logo_correct_size/data4_labelled.hdf5', 'r+')
base_items = list(f.items())
dset = f.get('labelled_data')
imgs = np.array(dset.get('observation'))
corn = np.array(dset.get('corners'))
observed_pos = 150
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
ORB_matches =cv.drawMatches(logo_temp, kp_temp, src_gray, kp_target, matches[15:300], None, flags=2)
plt.imshow(ORB_matches),plt.show()

#plotting the bounding boxes of detection and ground truth:
#1) finding the diagonal points (if recorded)
# cur_corn = corn[observed_pos,:,:]
# dists = []
# for i in range(0,4):
#     rest_corn = np.delete(corn[observed_pos,:,:],i,0)
#     for j in range(0,3):
#         if all(cur_corn[i,:] != [0,0]) and all(rest_corn[j,:] != [0,0]):
#             diff = np.sum(abs(np.array(cur_corn[i,:])-np.array(rest_corn[j,:])))
#             dists.append(diff)
#         else:
#             dists.append(0)
# # print(max(dists))
# diag = max(dists)
# for k in range(0,len(dists)):
#     if dists[k] == diag:
#         i = (k-np.mod(k,3))/3
#         corn1 = cur_corn[i,:]
#         rest_corn = np.delete(corn[observed_pos,:,:],i,0)
#         corn2 = rest_corn[np.mod(k,3),:]
#         break
# img = imgs[observed_pos,:,:,:]
# cv.rectangle(img,(corn1[1],corn1[0]),(corn2[1],corn2[0]),(1,0,0),1)
# plt.imshow(img),plt.show()

#locating keypoints on template in 3D coordinates
size = logo_temp.shape
print(size)
vert_ratio = 0.985/size[0]
horiz_ratio = 0.68/size[1]
x_origin = round(size[0]/2)
y_origin = round(size[1]/2)
pt_origin = np.array([int(x_origin), int(y_origin)])
print(pt_origin)
pos3d = []
for m in range(10,300):
    pt_idx = matches[0].imgIdx
    temp_coord = kp_temp[pt_idx].pt
    rel_pos_px = pt_origin - np.array(temp_coord)
    rel_pos_3d = np.array([rel_pos_px[0]/vert_ratio,rel_pos_px[1]/horiz_ratio,0.001])
    pos3d.append(rel_pos_3d)
print(len(pos3d))
    
    


        


