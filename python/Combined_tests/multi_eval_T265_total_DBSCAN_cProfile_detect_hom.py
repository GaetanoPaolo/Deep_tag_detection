import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
import h5py
import crop
import transform_mat as tm
import detect_match_opt_cuda as det
import draw_transf as dw
import cProfile
from sklearn.cluster import DBSCAN
import time
import concurrent.futures
#load the logo templates corresponding to different percentages of kept
#resolution from the original logo
logo_temp_1 = cv.imread('/home/gaetan/code/simulation_ws/src/my_simulations/models/psi_logo/materials/textures/poster-psi-drone-logo-1percent.png',0)
logo_temp_1 = crop.crop_img(logo_temp_1,1)
logo_temp_2 = cv.imread('/home/gaetan/code/simulation_ws/src/my_simulations/models/psi_logo/materials/textures/poster-psi-drone-logo-2percent.png',0)
logo_temp_2 = crop.crop_img(logo_temp_2,2)
logo_temp_5 = cv.imread('/home/gaetan/code/simulation_ws/src/my_simulations/models/psi_logo/materials/textures/poster-psi-drone-logo-5percent.png',0)
logo_temp_5 = crop.crop_img(logo_temp_5,5)
logo_temp_10 = cv.imread('/home/gaetan/code/simulation_ws/src/my_simulations/models/psi_logo/materials/textures/poster-psi-drone-logo-10percent.png',0)
logo_temp_10 = crop.crop_img(logo_temp_10,10)

# load the camera parameters stored in episode 1
f = h5py.File('/home/gaetan/data/hdf5/T265/data4_sync.hdf5', 'r+')
base_items = list(f.items())
dset1 = f.get('1')
K = np.array(dset1.get('K')).reshape((3,3))

# Load other parameters and images from chosen episode
# dset2 = f.get('1')
# group_items = list(dset2.items())
# imgs = np.array(dset2.get('observation'))
# pos = np.array(dset2.get('position'))
# quat = np.array(dset2.get('orientation'))
# rel_pos = np.array(dset2.get('relative_position'))
# img_stamp = np.array(dset2.get('image_time'))
# pose_stamp = np.array(dset2.get('pose_time'))
# set_size = imgs.shape

#defining the different detectors and matchers
orb = cv.ORB_create(2000,1.1,8,21,0,2,0,21,20)
sift = cv.SIFT_create(2000,6,0.02,30,1.6,cv.CV_32F)
brisk = cv.BRISK_create(3,6,0.9)
bf_HAMMING = cv.BFMatcher_create(cv.NORM_HAMMING,crossCheck=True)
bf_L2= cv.BFMatcher_create(cv.NORM_L2,crossCheck=True)

#a priori detection of keypoints on the templates using the different methods
kp_temp_orb_1, des_temp_orb_1 = orb.detectAndCompute(logo_temp_1,None)
kp_temp_orb_2, des_temp_orb_2 = orb.detectAndCompute(logo_temp_2,None)
kp_temp_orb_5, des_temp_orb_5 = orb.detectAndCompute(logo_temp_5,None)
kp_temp_orb_10, des_temp_orb_10 = orb.detectAndCompute(logo_temp_10,None)

#defining the function containing all the processing steps
def pose_est(kp_temp_orb_1, des_temp_orb_1,kp_temp_orb_2, des_temp_orb_2,
             kp_temp_orb_5, des_temp_orb_5,kp_temp_orb_10, des_temp_orb_10,
             logo_temp_1_shape,logo_temp_2_shape,logo_temp_5_shape,logo_temp_10_shape,
             K,orb,bf_HAMMING,src_gray):
    if rel_pos_c[2] < 3.0:
        try:
            zerarr = np.array([[0.0],
                                [0.0],
                                [0.0]])
            #computing ORB estimates for all different resolution percentages
            src_gray_2 = src_gray[144:657,144:705]
            kp_target, des_target = orb.detectAndCompute(src_gray_2,None)

            #https://learnopencv.com/getting-started-opencv-cuda-module/
            #https://docs.opencv.org/4.x/d5/dc3/group__cudalegacy.html
            #https://developer.nvidia.com/cuda-gpus
            pos_temp_1,pos_target_1, inlier_am_1,mask_1= det.detect_match(K,kp_temp_orb_1,des_temp_orb_1,logo_temp_1_shape,kp_target,des_target,bf_HAMMING,src_gray)
            pos_temp_2,pos_target_2, inlier_am_2,mask_2= det.detect_match(K,kp_temp_orb_2,des_temp_orb_2,logo_temp_2_shape,kp_target,des_target,bf_HAMMING,src_gray)
            pos_temp_5,pos_target_5, inlier_am_5,mask_5= det.detect_match(K,kp_temp_orb_5,des_temp_orb_5,logo_temp_5_shape,kp_target,des_target,bf_HAMMING,src_gray)
            pos_temp_10,pos_target_10,inlier_am_10,mask_10 = det.detect_match(K,kp_temp_orb_10,des_temp_orb_10,logo_temp_10_shape,kp_target,des_target,bf_HAMMING,src_gray)
            mask_lst = [mask_1,mask_2,mask_5,mask_10]
            inlier_am_lst = [inlier_am_1,inlier_am_2,inlier_am_5,inlier_am_10]
            temp_lst = [pos_temp_1,pos_temp_2,pos_temp_5,pos_temp_1,pos_temp_10]
            target_lst = [pos_target_1,pos_target_2,pos_target_5,pos_target_1,pos_target_10]
            shapes = [logo_temp_1_shape,logo_temp_2_shape,logo_temp_5_shape,logo_temp_10_shape]
            if sum(inlier_am_lst) == 0:
                trans = zerarr
            else:
                idx = inlier_am_lst.index(max(inlier_am_lst))
                if inlier_am_lst[idx] < 4:
                    trans = zerarr
                else:
                    pos_temp = temp_lst[idx]
                    pos_target = target_lst[idx]
                    cur_mask = mask_lst[idx]
                    #selecting which orb estimate is closest to the altitude estimated on drone 
                    # (assuming this provides a measure of the general accuracy)
                    # est_sift_res = det.resolution_sel(est_sift_lst,rel_pos_c)
                    #est_orb_res = det.resolution_sel(est_orb_lst,rel_pos_c)
                    # trans_est_sift.append(list(est_sift_res.astype(np.float64)))
                    #apply solvepnp to estimate translation: https://learnopencv.com/head-pose-estimation-using-opencv-and-dlib/
                    #try:
                    pos_temp_inlier = []
                    pos_target_inlier = []
                    for i in range(0,len(cur_mask)):
                        if cur_mask[i] == 1:
                            pos_temp_inlier.append(pos_temp[i])
                            pos_target_inlier.append(pos_target[i])
                    pos_temp_world = dw.world_coord(np.array(pos_temp_inlier),shapes[idx],0)
                    dist_coeffs = np.zeros((4,1))
                    (suc,rot,trans) = cv.solvePnP(pos_temp_world, np.array(pos_target_inlier), K, dist_coeffs, flags=cv.SOLVEPNP_ITERATIVE)
                #     return zerarr,[]
        except:
            trans = zerarr
        est_orb_res = np.append(trans,np.zeros((1,1)),axis = 0)
    else:
        try:
            temp_size = logo_temp_2_shape
            src_gray = src_gray[144:657,144:705]
            kp_target, des_target = orb.detectAndCompute(src_gray,None)
            #matching keypoints 
            matches = bf_HAMMING.match(des_temp_orb_2, des_target)
            #import pdb; pdb.set_trace()
            matches = sorted(matches,key=lambda x:x.distance)
            # creating keypoint list and template corner list
            corn_temp = [[0,0],[temp_size[1],0],[temp_size[1],temp_size[0]],[temp_size[1],0]]
            pos_temp = []
            pos_target = []
            match_target = []
            rot = 0
            for m in range(0,len(matches)):
                target_pt_idx = matches[m].trainIdx
                temp_pt_idx = matches[m].queryIdx
                temp_coord = kp_temp_orb_2[temp_pt_idx].pt
                target_coord = kp_target[target_pt_idx].pt
                match_target.append(kp_target[target_pt_idx])
                pos_temp.append(temp_coord)
                pos_target.append(target_coord)

            inlier_kp_target = pos_target
            #if less than 20 and at least two inliers: activation of the clustering algorithm to add edge matches
            #if not isinstance(inliers,type(None)):
            if len(matches) >= 2 and rel_pos_c[2] > 3:
                #try: 
                # creating keypoint list and template corner list
                kp_lst = []
                for m in range(0,len(kp_target)):
                    cur_pt = kp_target[m].pt
                    int_pt = [int(cur_pt[0]),int(cur_pt[1])]
                    kp_lst.append(int_pt)
                    corn_temp = [[0,0],[temp_size[1],0],[temp_size[1],temp_size[0]],[0,temp_size[0]]]
                #calculating the mean distance between all matches
                match_arr = det.kp_preproc(match_target)
                match_len = match_arr.shape
                dist_norms = np.zeros((match_len[0]**2,))
                for k in range(1,match_len[0]):
                    cur_shift = np.roll(match_arr,k,axis = 0)
                    diff = np.subtract(match_arr,cur_shift)
                    cur_norm_set = np.linalg.norm(diff,axis = 1)
                    dist_norms[match_len[0]*(k-1):match_len[0]*k] = cur_norm_set
                
                dist_param = (np.mean(dist_norms, axis = 0)/(0.002*len(kp_target)))+10

                #finding the corner points by fitting a rectangle around the clustered keypoints
                #https://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html#sphx-glr-auto-examples-cluster-plot-dbscan-py
                #https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html
                Z = det.kp_preproc(kp_target)
                # apply DBSCAN
                db = DBSCAN(eps=dist_param, min_samples=5).fit(Z)
                core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
                core_samples_mask[db.core_sample_indices_] = True
                labels = db.labels_

                # #############################################################################

                # Selecting cluster as the one that contains the larges amount of matches
                unique_labels = set(labels)
                match_counters = []
                for k in unique_labels:
                    class_member_mask = labels == k
                    xy = Z[class_member_mask & core_samples_mask]
                    match_counter = 0
                    for i in range(0,len(inlier_kp_target)):
                        inlier_match_arr = np.array(inlier_kp_target[i])
                        if inlier_match_arr in xy:
                            match_counter += 1
                    match_counters.append(match_counter)
                # find which class contains the matches
                class_pos = match_counters.index(max(match_counters))
                class_member_mask = labels==class_pos
                A = Z[class_member_mask  & core_samples_mask]
                #computing bounding box around the selected cluster
                corn_target = dw.findBB_rot(A)
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
                n = 3
                rot_step = 2**n
                point_amount = 4*rot_step
                ordered_pts = det.rectangle_side_fractal(ordered_pts,point_amount)
                corn_temp = det.rectangle_side_fractal(corn_temp,point_amount)

                #checking the mean keypoint distances relative to the centers of the long sides
                #long sides : 6, 14
                # short sides : 2, 10
                # for varying values of n: 
                #long sides: rot_step + 2**(n-1), 3*rot_step + 2**(n-1)
                #short sides: 2**(n-1), 2*rot_step + 2**(n-1)
                idx1 = rot_step + 2**(n-1)
                idx2 = 3*rot_step + 2**(n-1)
                pt1 = corn_temp[idx1]
                pt2 = corn_temp[idx2]
                dst1 = det.mean_keypoint_dist_temp(kp_temp_orb_2,pt1)
                dst2 = det.mean_keypoint_dist_temp(kp_temp_orb_2,pt2)
                dst_temp = np.array([dst1,dst2])
                # print('dst_temp')
                # print(dst_temp)
                rot = 0
                pos_temp =  corn_temp + pos_temp
                dist_coeffs = np.array([-0.014216,0.060412,-0.054711,0.011151])
                candidate_trans = []
                inlier_arr = [0]
                pts_arr = []
                for shift in range(0,4):
                    ordered_pts_2 = list(np.roll(np.array(ordered_pts),-rot_step*shift,axis = 0))
                    pos_temp_world = dw.world_coord(np.array(pos_temp),logo_temp_2_shape,rot)
                    pos_target_2 = ordered_pts_2 + pos_target
                    (suc,angle,trans,inliers) = cv.solvePnPRansac(pos_temp_world, np.array(pos_target_2), K, dist_coeffs, flags=cv.SOLVEPNP_ITERATIVE, iterationsCount=2000, reprojectionError=8.0)
                    if inliers is not None:
                        if len(inliers) >= max(inlier_arr):
                            candidate_trans.append(trans)
                            pts_arr.append(ordered_pts_2)
                        inlier_arr.append(len(inliers))
                kp_dist_diff = []
                if len(pts_arr) == 0:
                    trans = np.zeros((3,1))
                else:
                    for k in range(0,len(pts_arr)):
                        pt1 = pts_arr[k][idx1]
                        pt2 = pts_arr[k][idx2]
                        dst1 = det.mean_keypoint_dist_target(A,pt1)
                        dst2 = det.mean_keypoint_dist_target(A,pt2)
                        dst_target = np.array([dst1,dst2])
                        diff = np.linalg.norm(np.subtract(dst_temp,dst_target))
                        kp_dist_diff.append(diff)
                    trans = candidate_trans[kp_dist_diff.index(min(kp_dist_diff))]
                est_orb_res = np.append(trans,np.zeros((1,1)),axis = 0)
                # print('Chosen rotation')
                # print(pts_arr[kp_dist_diff.index(min(kp_dist_diff))])
            else:
                trans = np.zeros((3,1))
                est_orb_res = np.append(trans,np.zeros((1,1)),axis = 0)
        except:
            trans = np.zeros((3,1))
            est_orb_res = np.append(trans,np.zeros((1,1)),axis = 0)
    return est_orb_res


#defining the arrays in which the results will be stored 
trans_est_orb = [] 
drone_est = []
trans_est_sift = []
tot_timing = []
#parsing the workable dataset for method evaluation
for ep in range(1,2):
    print(ep)
    ep_str = "% s" % ep
    dset2 = f.get(ep_str)
    group_items = list(dset2.items())
    imgs = np.array(dset2.get('observation'))
    pos = np.array(dset2.get('position'))
    quat = np.array(dset2.get('orientation'))
    rel_pos = np.array(dset2.get('relative_position'))
    img_stamp = np.array(dset2.get('image_time'))
    pose_stamp = np.array(dset2.get('pose_time'))
    set_size = imgs.shape
    #for observed_pos in range(0,set_size[0]):
    for observed_pos in range(190,191):
        src = imgs[observed_pos,:,:,:]*255
        src_gray = np.uint8(src)
        #computing the estimated tag position relative to the camera (from drone world axes position estimate)
        rel_pos_w = -rel_pos[observed_pos,:]
        #Transforming from world axes to axes attached to the drone
        T_w_d = tm.transf_mat(quat[observed_pos,:],np.zeros((3,1)))
        R_w_d = T_w_d[0:3,0:3]
        #Transforming from axes attached to the drone to camera axes
        R_d_c = np.matrix([[0,1,0],
                    [1,0,0],
                    [0,0,-1]])
        rel_pos_d = np.matmul(R_w_d,np.array(rel_pos_w))
        rel_pos_c = np.matmul(R_d_c,np.transpose(rel_pos_d))
        # print('GT approx')
        # print(rel_pos_c)
        drone_est.append(list(rel_pos_c.astype(np.float64)))
        cProfile.run("pose_est(kp_temp_orb_1, des_temp_orb_1,kp_temp_orb_2, des_temp_orb_2, kp_temp_orb_5, des_temp_orb_5,kp_temp_orb_10, des_temp_orb_10,logo_temp_1.shape,logo_temp_2.shape,logo_temp_5.shape,logo_temp_10.shape, K,orb,bf_HAMMING,src_gray)",
                    "detect_parallel_hom_pose_est_below_3m_190.dat")
        # start = time.time()
        # est_orb_res = pose_est(kp_temp_orb_1, des_temp_orb_1,kp_temp_orb_2, des_temp_orb_2, kp_temp_orb_5, des_temp_orb_5,kp_temp_orb_10, des_temp_orb_10,logo_temp_1.shape,logo_temp_2.shape,logo_temp_5.shape,logo_temp_10.shape, K,orb,bf_HAMMING,src_gray)
        # end = time.time()
        # timing = end - start
        # tot_timing.append(timing)
        # trans_est_orb.append(list(est_orb_res.astype(np.float64)))

# hdf5_data = {"trans_est_orb": trans_est_orb,"drone_est":drone_est, 'timing':tot_timing}
# #exporting the computations to hdf5
# current_dir = '/home/gaetan/data/hdf5/'
# def dump(output_dir,hdf5_data,ep):
#         print('stored data in',output_dir)
#         output_hdf5_path = output_dir + '/T265_alt_DBSCAN_8repr_clust_solve_naive_timing_detect_hom' + '.hdf5'
#         hdf5_file = h5py.File(output_hdf5_path, "a")
#         episode_group = hdf5_file.create_group(str(ep))
#         for sensor_name in hdf5_data.keys():
#             episode_group.create_dataset(
#                 sensor_name, data=np.stack(hdf5_data[sensor_name])
#             )
#         hdf5_file.close()
# dump(current_dir,hdf5_data,'T265_alt_DBSCAN_8repr_clust_solve_naive_timing_detect_hom')



