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
from collections import deque
from sklearn.cluster import DBSCAN
from sklearn import metrics


#load the logo template
logo_temp = cv.imread('/home/gaetan/code/simulation_ws/src/my_simulations/models/psi_logo/materials/textures/poster-psi-drone-logo-2percent.png',0)
temp_size = logo_temp.shape
rot = 0

#removing the blue edge of the logo template to detect logo itself
#logo_temp = crop.crop_img(logo_temp,2)
plt.imshow(logo_temp),plt.show()
# load the labelled gazebo data
f = h5py.File('/home/gaetan/data/hdf5/psi_800res_alt_rot_qr/data4_sync.hdf5', 'r+')
base_items = list(f.items())
print(base_items)
dset = f.get('15')
imgs = np.array(dset.get('observation'))
corn = np.array(dset.get('corners'))
pos = np.array(dset.get('position'))
pos_origin_cam = np.array(dset.get('pos_origin_cam'))
print('image amount')
print(imgs.shape)
observed_pos = 0
src = imgs[observed_pos,:,:,:]*255
src_gray = np.uint8(cv.cvtColor(src, cv.COLOR_BGR2GRAY))
plt.imshow(src_gray),plt.show()

#code source till line 31: https://datahacker.rs/feature-matching-methods-comparison-in-opencv/
#creating keypoint matchers and finders
orb = cv.ORB_create(2000,1.1,8,21,0,2,0,21,20)
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
print(len(kp_target))
src_gray_2 = cv.drawKeypoints(src_gray,kp_target,0,(255,0,0),cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
plt.imshow(src_gray_2),plt.show()

#matching keypoints 
matches = bf.match(des_temp, des_target)
#import pdb; pdb.set_trace()
matches = sorted(matches,key=lambda x:x.distance)
print('Match amount')
print(len(matches))
ORB_matches =cv.drawMatches(logo_temp, kp_temp, src_gray, kp_target, matches[0:len(matches)], None, flags=2)
plt.imshow(ORB_matches),plt.show()

pos_temp = []
pos_target = []
match_target = []
#2) approximating camera parameters, will be approximated again during solvepnp
# if result incorrect => calibrate camera_
# fx = 119
# fy = 119
horizontal_field_of_view = (80 * img_size[1]/img_size[0]) * 3.14 / 180
vertical_field_of_view = 80 * 3.14 / 180
fx = img_size[1]/2*np.tan(horizontal_field_of_view/2)**(-1)
fy = img_size[0]/2*np.tan(vertical_field_of_view/2)**(-1)
K = np.array([[fx,0,img_size[1]/2],
                [0,fy,img_size[0]/2],
                [0,0,1]])

for m in range(0,len(matches)):
    target_pt_idx = matches[m].trainIdx
    temp_pt_idx = matches[m].queryIdx
    temp_coord = kp_temp[temp_pt_idx].pt
    target_coord = kp_target[target_pt_idx].pt
    match_target.append(kp_target[target_pt_idx])
    pos_temp.append(temp_coord)
    pos_target.append(target_coord)
print('Current alt')
print(pos_origin_cam[observed_pos,2])

#apply solvepnp to estimate translation: https://learnopencv.com/head-pose-estimation-using-opencv-and-dlib/
pos_temp_world = dw.world_coord(np.array(pos_temp),logo_temp,rot)
dist_coeffs = np.zeros((4,1))
(suc,angle,trans,inliers) = cv.solvePnPRansac(pos_temp_world, np.array(pos_target), K, dist_coeffs, flags=cv.SOLVEPNP_ITERATIVE, iterationsCount=2000, reprojectionError=2.0)

#plotting the inlier matches if present
print('Inlier match amount')
if isinstance(inliers,type(None)):
    print(inliers)
else:
    print(len(inliers))
    inlier_matches = []
    inlier_kp = []
    for k in range(0,len(inliers)):
        inlier_matches.append(matches[inliers[k][0]])
        cur_inlier_idx = matches[inliers[k][0]].trainIdx
        inlier_kp.append(kp_target[cur_inlier_idx])
    ORB_matches =cv.drawMatches(logo_temp, kp_temp, src_gray, kp_target, inlier_matches, None, flags=2)
    plt.imshow(ORB_matches),plt.show()
inlier_kp_target = pos_target
#if less than 20 and at least two inliers: activation of the clustering algorithm to add edge matches
#if not isinstance(inliers,type(None)):
if len(matches) >= 2 and pos_origin_cam[observed_pos,2] > 3:
    # creating keypoint list and template corner list
    kp_lst = []
    for m in range(0,len(kp_target)):
        cur_pt = kp_target[m].pt
        int_pt = [int(cur_pt[0]),int(cur_pt[1])]
        kp_lst.append(int_pt)
        corn_temp = [[0,0],[temp_size[1],0],[temp_size[1],temp_size[0]],[0,temp_size[0]]]
    
    #calculating the mean distance between all matches
    match_arr = dm.kp_preproc(match_target)
    match_len = match_arr.shape
    dist_norms = np.zeros((match_len[0]**2,))
    for k in range(1,match_len[0]):
        cur_shift = np.roll(match_arr,k,axis = 0)
        diff = np.subtract(match_arr,cur_shift)
        cur_norm_set = np.linalg.norm(diff,axis = 1)
        dist_norms[match_len[0]*(k-1):match_len[0]*k] = cur_norm_set
    
    dist_param = np.mean(dist_norms, axis = 0)/(0.002*len(kp_target))
    print('DBSCAN distance')
    print(dist_param +10)


    #finding the corner points by fitting a rectangle around the clustered keypoints
    #https://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html#sphx-glr-auto-examples-cluster-plot-dbscan-py
    #https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html
    Z = dm.kp_preproc(kp_target)
    print('kp before clustering')
    print(Z.shape)
    # apply DBSCAN
    db = DBSCAN(eps=dist_param +10, min_samples=5).fit(Z)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    # #############################################################################
    # Plot result
    import matplotlib.pyplot as plt

    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    match_counters = []
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = labels == k
        xy = Z[class_member_mask & core_samples_mask]
        match_counter = 0
        for i in range(0,len(inlier_kp_target)):
            inlier_match_arr = np.array(inlier_kp_target[i])
            if inlier_match_arr in xy:
                match_counter += 1
        match_counters.append(match_counter)
        plt.plot(
            xy[:, 0],
            xy[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=14,
        )

        xy = Z[class_member_mask & ~core_samples_mask]
        plt.plot(
            xy[:, 0],
            xy[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=6,
        )
    plt.title("Estimated number of clusters: %d" % n_clusters_)
    plt.show()
    # find which class contains the matches
    class_pos = match_counters.index(max(match_counters))
    class_member_mask = labels==class_pos
    A = Z[class_member_mask  & core_samples_mask]
    plt.plot(
            A[:, 0],
            A[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=14,
            )
    plt.show()
    
    corn_target = dw.findBB_rot(A)

    # #plotting the obtained bounding box points
    img_cont1 = np.copy(src_gray)
    cv.drawContours(img_cont1,[corn_target],0,(255,0,0),1)
    plt.imshow(img_cont1),plt.show()
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
    n = 3
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
    idx1 = rot_step + 2**(n-1)
    idx2 = 3*rot_step + 2**(n-1)
    print('idx1')
    print(idx1)
    print('idx2')
    print(idx2)
    pt1 = corn_temp[idx1]
    pt2 = corn_temp[idx2]
    dst1 = dm.mean_keypoint_dist_temp(kp_temp,pt1)
    dst2 = dm.mean_keypoint_dist_temp(kp_temp,pt2)
    dst_temp = np.array([dst1,dst2])
    print('dst_temp')
    print(dst_temp)

    
    pos_temp =  corn_temp + pos_temp
    pos_temp =  corn_temp
    dist_coeffs = np.zeros((4,1))
    candidate_trans = []
    inlier_arr = [0]
    pts_arr = []
    for shift in range(0,4):
        ordered_pts_2 = list(np.roll(np.array(ordered_pts),-rot_step*shift,axis = 0))
        pos_temp_world = dw.world_coord(np.array(pos_temp),logo_temp,rot)
        pos_target_2 = ordered_pts_2 + pos_target
        pos_target_2 = ordered_pts_2
        (suc,angle,trans,inliers) = cv.solvePnPRansac(pos_temp_world, np.array(pos_target_2), K, dist_coeffs, flags=cv.SOLVEPNP_ITERATIVE, iterationsCount=2000, reprojectionError=2.0)
        if inliers is not None:
            if len(inliers) >= max(inlier_arr):
                candidate_trans.append(trans)
                pts_arr.append(ordered_pts_2)
            inlier_arr.append(len(inliers))
    print('Inlier arr')
    print(inlier_arr)
    kp_dist_diff = []
    for k in range(0,len(pts_arr)):
        pt1 = pts_arr[k][idx1]
        pt2 = pts_arr[k][idx2]
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
    print(kp_dist_diff.index(min(kp_dist_diff)))
    print(pts_arr[kp_dist_diff.index(min(kp_dist_diff))])




print('Estimate origin pose in camera coord')
print(trans)


#computing the error between estimated relative pose and GT
print('GT in camera coordinates')
print(pos_origin_cam[observed_pos,:])
diff = np.subtract(pos_origin_cam[observed_pos,0:3],trans[0:3])
print('Diff')
print(diff)
#print("GT range")
#print(pos_origin_cam[observed_pos-10:observed_pos+10,0:3])



        


