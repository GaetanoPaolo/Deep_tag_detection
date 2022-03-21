from matplotlib import pyplot as plt
import cv2 as cv
import os
import numpy as np
import h5py
import time
import crop
import itertools
import draw_transf as dw
import detect_match as dm
#load the logo template
logo_temp = cv.imread('/home/gaetan/code/simulation_ws/src/my_simulations/models/coca_cola_logo/materials/textures/cocacola_15perc.jpg',0)
rot = 0

#removing the blue edge of the logo template to detect logo itself
#logo_temp = crop.crop_img(logo_temp,2)
plt.imshow(logo_temp),plt.show()
# load the labelled gazebo data
f = h5py.File('/home/gaetan/data/hdf5/coca_cola_800res_alt_rot/data4_sync.hdf5', 'r+')
base_items = list(f.items())
print(f.items)
dset = f.get('10')
imgs = np.array(dset.get('observation'))
corn = np.array(dset.get('corners'))
pos = np.array(dset.get('position'))
pos_origin_cam = np.array(dset.get('pos_origin_cam'))
observed_pos =48
src = imgs[observed_pos,:,:,:]*255
src_gray = np.uint8(cv.cvtColor(src, cv.COLOR_BGR2GRAY))
plt.imshow(src_gray),plt.show()

#code source till line 31: https://datahacker.rs/feature-matching-methods-comparison-in-opencv/
#creating keypoint matchers and finders
orb = cv.ORB_create(500,1.1,8,21,0,2,21,20)
bf = cv.BFMatcher_create(cv.NORM_HAMMING,crossCheck=True)

#creating mask where keypoints have to be found
cur_corn = corn[observed_pos,:,:]
corn_size = cur_corn.shape
img_size = src_gray.shape
temp_size = logo_temp.shape
#calculating keypoints and finding the matching ones
kp_temp, des_temp = orb.detectAndCompute(logo_temp,None)
print(len(kp_temp))
logo_temp_2 = cv.drawKeypoints(logo_temp,kp_temp,0,(255,0,0),cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
plt.imshow(logo_temp_2),plt.show()
kp_target, des_target = orb.detectAndCompute(src_gray,None)
print(len(kp_target))
src_gray_2 = cv.drawKeypoints(src_gray,kp_target,0,(255,0,0),cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
plt.imshow(src_gray_2),plt.show()
# creating keypoint list and template corner list
kp_lst = []
for m in range(0,len(kp_target)):
    cur_pt = kp_target[m].pt
    int_pt = [int(cur_pt[0]),int(cur_pt[1])]
    kp_lst.append(int_pt)
corn_temp = [[0,0],[temp_size[1],0],[temp_size[1],temp_size[0]],[0,temp_size[0]]]


#finding the corner points by fitting a rectangle around the clustered keypoints
#https://docs.opencv.org/4.x/d1/d5c/tutorial_py_kmeans_opencv.html
Z = dm.kp_preproc(kp_target)
# define criteria and apply kmeans()
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
ret,label,center=cv.kmeans(Z,1,None,criteria,10,cv.KMEANS_RANDOM_CENTERS)
# Now separate the data, Note the flatten()
A = Z[label.ravel()==0]
corn_target = dw.findBB_rot(A)
#B = Z[label.ravel()==1]
# Plot the data
plt.scatter(A[:,0],A[:,1])
#plt.scatter(B[:,0],B[:,1],c = 'r')
plt.scatter(center[:,0],center[:,1],s = 80,c = 'y', marker = 's')
plt.xlabel('Height'),plt.ylabel('Weight')
plt.show()


#rearranging BB points in clockwise direction
# https://pyimagesearch.com/2016/03/21/ordering-coordinates-clockwise-with-python-and-opencv/ 
ordered_pts = np.zeros((4,2))
s = np.sum(corn_target, axis = 1)
ordered_pts[0,:] = corn_target[np.argmin(s)]
ordered_pts[2,:] = corn_target[np.argmax(s)]
diff = np.diff(corn_target, axis = 1)
ordered_pts[1,:] = corn_target[np.argmin(diff)]
ordered_pts[3,:] = corn_target[np.argmax(diff)]
ordered_pts = list(ordered_pts)


#expand list with middle point of each side of the rectangle
#this expansion happens with a factor 2 at every recursion
#The amount of rotations required later to test the different projections
#is also dependent on this amount of recursions.
#0 recursion/1 execution = 8 points in total = 2 rotations
#1 recursion/2 executions = 16 points in total = 4 rotation
#...
#=> n executions = 2^n rotations = 4*2^n points in total
#After inspection, this setup maintains symmetry => still degree of freedom around the diagonal
n = 2
rot_step = 2**n
point_amount = 4*rot_step
ordered_pts = dm.rectangle_side_fractal(ordered_pts,point_amount)
corn_temp = dm.rectangle_side_fractal(corn_temp,point_amount)

#checking the mean keypoint distances relative to the centers of the long sides
#long sides : 6, 14
# short sides : 2, 10
# for varying values of n: 
#long sides: rot_step + 2**(n-1), 3*rot_step + 2**(n-1)
#short sides: 2**(n-1), 2*rot_step + 2**(n-1)
pt1 = corn_temp[6]
pt2 = corn_temp[14]
dst1 = dm.mean_keypoint_dist_temp(kp_temp,pt1)
dst2 = dm.mean_keypoint_dist_temp(kp_temp,pt2)
dst_temp = np.array([dst1,dst2])
print('dst_temp')
print(dst_temp)
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
if pos_origin_cam[observed_pos,2] > 3:
    pos_temp =  corn_temp
    dist_coeffs = np.zeros((4,1))
    candidate_trans = []
    inlier_arr = [0]
    pts_arr = []
    for shift in range(0,4):
        ordered_pts_2 = list(np.roll(np.array(ordered_pts),-rot_step*shift,axis = 0))
        pos_temp_world = dw.world_coord(np.array(pos_temp),logo_temp,rot)
        pos_target = ordered_pts_2
        (suc,angle,trans,inliers) = cv.solvePnPRansac(pos_temp_world, np.array(pos_target), K, dist_coeffs, flags=cv.SOLVEPNP_ITERATIVE, iterationsCount=2000, reprojectionError=2.0)
        if inliers is not None:
            if len(inliers) >= max(inlier_arr):
                candidate_trans.append(trans)
                pts_arr.append(ordered_pts_2)
            inlier_arr.append(len(inliers))
    kp_dist_diff = []
    for k in range(0,len(pts_arr)):
        pt1 = pts_arr[k][6]
        pt2 = pts_arr[k][14]
        dst1 = dm.mean_keypoint_dist_target(A,pt1)
        dst2 = dm.mean_keypoint_dist_target(A,pt2)
        dst_target = np.array([dst1,dst2])
        print('dst_target')
        print(k)
        print(dst_target)
        diff = np.linalg.norm(np.subtract(dst_temp,dst_target))
        kp_dist_diff.append(diff)
    


    trans = candidate_trans[kp_dist_diff.index(min(kp_dist_diff))]
    print('Chosen rotation')
    print(pts_arr[kp_dist_diff.index(min(kp_dist_diff))])
else:
    #apply solvepnp to estimate translation: https://learnopencv.com/head-pose-estimation-using-opencv-and-dlib/
    pos_temp_world = dw.world_coord(np.array(pos_temp),logo_temp,rot)
    dist_coeffs = np.zeros((4,1))
    (suc,rot,trans,inliers) = cv.solvePnPRansac(pos_temp_world, np.array(pos_target), K, dist_coeffs, flags=cv.SOLVEPNP_ITERATIVE, iterationsCount=2000, reprojectionError=8)
    # [H,mask] = cv.findHomography(pos_temp_world, np.array(pos_target),cv.RANSAC,2.0,2000)
    # [retval,rot,trans,normals] = cv.decomposeHomographyMat(H, K)
    print('Inlier match amount')
    print(len(inliers))
    inlier_matches = []
    for k in range(0,len(inliers)):
        inlier_matches.append(matches[inliers[k][0]])
    ORB_matches =cv.drawMatches(logo_temp, kp_temp, src_gray, kp_target, inlier_matches, None, flags=2)


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
plt.imshow(ORB_matches),plt.show() 


        


