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
import detect_match as dm
#load the logo template
logo_temp = cv.imread('/home/gaetan/code/simulation_ws/src/my_simulations/models/psi_logo/materials/textures/poster-psi-drone-logo-2percent.png',0)
logo_temp_color = cv.imread('/home/gaetan/code/simulation_ws/src/my_simulations/models/psi_logo/materials/textures/poster-psi-drone-logo.png')
#The scale var indicates the percentage kept from the original logo resolution
#scale = 2
rot = 0
plt.imshow(logo_temp),plt.show()
#logo_temp = crop.crop_img(logo_temp,scale)
# load the camera parameters stored in episode 1
f = h5py.File('/home/gaetan/data/hdf5/T265/data4_sync.hdf5', 'r+')
base_items = list(f.items())
print(base_items)
dset1 = f.get('1')
K = np.array(dset1.get('K')).reshape((3,3))
# ILoad other parameters and images from chosen episode
dset2 = f.get('1')
group_items = list(dset2.items())
imgs = np.array(dset2.get('observation'))
pos = np.array(dset2.get('position'))
quat = np.array(dset2.get('orientation'))
rel_pos = np.array(dset2.get('relative_position'))
img_stamp = np.array(dset2.get('image_time'))
pose_stamp = np.array(dset2.get('pose_time'))
observed_pos =295
src = imgs[observed_pos,:,:,:]*255
src_gray = np.uint8(src)
plt.imshow(imgs[observed_pos,:,:,:]),plt.show()
print(len(imgs))

#code source till line 31: https://datahacker.rs/feature-matching-methods-comparison-in-opencv/
#creating keypoint matchers and finders
#code source till line 31: https://datahacker.rs/feature-matching-methods-comparison-in-opencv/
#creating keypoint matchers and finders
#orb = cv.ORB_create(2000,1.1,8,11,0,2,0,11,5)
orb = cv.ORB_create(2000,1.1,8,21,0,2,0,21,20)
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

#converting the relative position to camera axes
#the given rel_pos (obtained by subtracting the starting position of the drone from the subsequent ones) is from the tag to the drone in world coordinates
#Converting to position from drone to tag:
print("Preproc rel pos")
print(rel_pos[observed_pos,:])
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

if rel_pos_c[2] > 3:

    #cluster the keypoints to try obtain the correct square
    #finding the corner points by fitting a rectangle around the clustered keypoints
    #https://docs.opencv.org/4.x/d1/d5c/tutorial_py_kmeans_opencv.html
    temp_size = logo_temp.shape
    corn_temp = np.array([[0,0],[temp_size[1],0],[temp_size[1],temp_size[0]],[0,temp_size[0]]])
    diag_temp = np.linalg.norm(np.array([temp_size[0],temp_size[1]]))
    temp_ratio = temp_size[1]/diag_temp
    Z = dm.kp_preproc(kp_target)
    # define criteria and apply kmeans()
    count = 1
    stop = False
    while stop == False:
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        ret,label,center=cv.kmeans(Z,count,None,criteria,10,cv.KMEANS_RANDOM_CENTERS)
        # Now separate the data, Note the flatten()
        for i in range(0,count):
            A = Z[label.ravel()==i]
            corn_target = dw.findBB_rot(A)
            dst = np.linalg.norm(np.subtract(corn_target,corn_target[0]),axis = 1)
            short_side = np.min(dst[1:3])
            diag = np.max(dst[1:3])
            ratio = short_side/diag
            ratio_diff = abs(ratio-temp_ratio)
            print(ratio_diff)
            if  ratio_diff < 0.3:
                stop = True
                break
        count += 1
    print('Corn target')
    print(corn_target)
    # corn_target = np.array([[346-144,408-144],
    #                         [376-144,408-144],
    #                         [376-144,450-144],
    #                         [346-144,450-144]])
    print('Corn temp')
    print(corn_temp)
    #plotting the obtained bounding box points
    img_cont1 = np.copy(src_gray)
    cv.drawContours(img_cont1,[corn_target],0,(255,0,0),1)
    plt.imshow(img_cont1),plt.show()

    #putting the BB corner coordinates back to the original camera image coordinates
    for m in range(0, len(corn_target)):
        corn_target[m,:] = np.array([corn_target[m,0]+144,corn_target[m,1]+144])
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
    pt1_sel = rot_step + 2**(n-1)
    pt2_sel = 3*rot_step + 2**(n-1)
    point_amount = 4*rot_step
    ordered_pts = dm.rectangle_side_fractal(ordered_pts,point_amount)
    corn_temp = dm.rectangle_side_fractal(corn_temp,point_amount)

    #checking the mean keypoint distances relative to the centers of the long sides
    #long sides : 6, 14
    # short sides : 2, 10
    # for varying values of n: 
    #long sides: rot_step + 2**(n-1), 3*rot_step + 2**(n-1)
    #short sides: 2**(n-1), 2*rot_step + 2**(n-1)
    pt1 = corn_temp[pt1_sel]
    pt2 = corn_temp[pt2_sel]
    dst1 = dm.mean_keypoint_dist_temp(kp_temp,pt1)
    dst2 = dm.mean_keypoint_dist_temp(kp_temp,pt2)
    dst_temp = np.array([dst1,dst2])
    print('dst_temp')
    print(dst_temp)

    #computing the projections for each of the possible shifts, 
    #choosing the two best permutations that are 180° shifted (by checking their inlier amount),
    #choosing the permutation that has 
    pos_temp =  corn_temp
    dist_coeffs = np.array([-0.014216,0.060412,-0.054711,0.011151])
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
    print('inlier arr')
    print(inlier_arr)
    
    kp_dist_diff = []
    for k in range(0,len(pts_arr)):
        pt1 = pts_arr[k][pt1_sel]
        pt2 = pts_arr[k][pt1_sel]
        dst1 = dm.mean_keypoint_dist_target(A,pt1)
        dst2 = dm.mean_keypoint_dist_target(A,pt2)
        dst_target = np.array([dst1,dst2])
        print(dst_target)
        diff = np.linalg.norm(np.subtract(dst_temp,dst_target))
        kp_dist_diff.append(diff)
    trans = candidate_trans[kp_dist_diff.index(min(kp_dist_diff))]
    print('Chosen rotation')
    print(pts_arr[kp_dist_diff.index(min(kp_dist_diff))]) 
    print(kp_dist_diff.index(min(kp_dist_diff)))
    print('Template points')
    print(corn_temp)
else:


    #match the keypoints
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
    dist_coeffs = np.array([-0.014216,0.060412,-0.054711,0.011151])
    (suc,rot,trans,inliers) = cv.solvePnPRansac(pos_temp_world, np.array(pos_target), K, dist_coeffs, flags=cv.SOLVEPNP_ITERATIVE, iterationsCount=2000, reprojectionError=2)
    print('Inlier match amount')
    print(len(inliers))
    inlier_matches = []
    for k in range(0,len(inliers)):
        inlier_matches.append(matches[inliers[k][0]])
    ORB_matches =cv.drawMatches(logo_temp, kp_temp, src_gray, kp_target, inlier_matches, None, flags=2)
    plt.imshow(ORB_matches),plt.show() 


print('Estimated pos relative to cam')
print(trans)
print('origin pos relative to cam')
print(rel_pos_c)
