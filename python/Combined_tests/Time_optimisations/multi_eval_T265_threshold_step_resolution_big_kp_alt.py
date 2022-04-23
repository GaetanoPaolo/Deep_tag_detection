from matplotlib import pyplot as plt
import cv2 as cv
import numpy as np
import h5py
import detect_match_opt_cuda as det
import draw_transf as dw
import transform_mat as tm
import cProfile
from sklearn.cluster import DBSCAN
import time
import solvepnp_gpu
#this code is a modification of:
#multi_eval_T265_total_DBSCAN_cProfile_detect_total_cuda_except_match_alt


#load the logo templates corresponding to different percentages of kept
#resolution from the original logo
logo_temp_1 = np.uint8(cv.imread('/home/gaetan/code/simulation_ws/src/my_simulations/models/psi_logo/materials/textures/poster-psi-drone-logo-1percent.png',0))
#logo_temp_1 = crop.crop_img(logo_temp_1,1)
logo_temp_2 = cv.imread('/home/gaetan/code/simulation_ws/src/my_simulations/models/psi_logo/materials/textures/poster-psi-drone-logo-2percent.png',0)
#logo_temp_2 = crop.crop_img(logo_temp_2,2)
logo_temp_5 = cv.imread('/home/gaetan/code/simulation_ws/src/my_simulations/models/psi_logo/materials/textures/poster-psi-drone-logo-5percent.png',0)
#logo_temp_5 = crop.crop_img(logo_temp_5,5)
logo_temp_10 = cv.imread('/home/gaetan/code/simulation_ws/src/my_simulations/models/psi_logo/materials/textures/poster-psi-drone-logo-10percent.png',0)
#logo_temp_10 = crop.crop_img(logo_temp_10,10)

#detect 3 central keypoints for  drone > 3m alg
curstom_kp_size = 16
orb_single = cv.ORB_create(2000,1.1,8,curstom_kp_size,0,2,0,curstom_kp_size,32)
print(logo_temp_2.shape)

central_kp_list_temp = [cv.KeyPoint(round(logo_temp_2.shape[1]/2),(logo_temp_2.shape[0]/2)+round(logo_temp_2.shape[0]/8),curstom_kp_size),
                        cv.KeyPoint(round(logo_temp_2.shape[1]/2),(logo_temp_2.shape[0]/2)-round(logo_temp_2.shape[0]/8),curstom_kp_size),
                        cv.KeyPoint(round(logo_temp_2.shape[1]/2),(logo_temp_2.shape[0]/2)+round(logo_temp_2.shape[0]/4),curstom_kp_size),
                        cv.KeyPoint(round(logo_temp_2.shape[1]/2),(logo_temp_2.shape[0]/2)-round(logo_temp_2.shape[0]/4),curstom_kp_size)]
central_kp_2_temp, central_kp_des_2 = orb_single.compute(logo_temp_2, central_kp_list_temp)

# src_gray_2 = cv.drawKeypoints(logo_temp_2,central_kp_2_temp,0,(255,0,0),cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# plt.imshow(src_gray_2),plt.show()
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
orb = cv.cuda.ORB_create(2000,1.1,8,21,0,2,0,21,5)
sift = cv.SIFT_create(2000,6,0.02,30,1.6,cv.CV_32F)
brisk = cv.BRISK_create(3,6,0.9)
bf_HAMMING = cv.BFMatcher_create(cv.NORM_HAMMING,crossCheck=True)
#bf_HAMMING = cv.cuda.DescriptorMatcher_createBFMatcher(cv.NORM_HAMMING)
bf_L2= cv.BFMatcher_create(cv.NORM_L2,crossCheck=True)
#a priori detection of keypoints on the templates using the different methods
#logo_temp_1_cuda = cv.cuda_GpuMat()
logo_temp_2_cuda = cv.cuda_GpuMat()
logo_temp_5_cuda = cv.cuda_GpuMat()
logo_temp_10_cuda = cv.cuda_GpuMat()
#logo_temp_1_cuda.upload(logo_temp_1)
logo_temp_2_cuda.upload(logo_temp_2)
logo_temp_5_cuda.upload(logo_temp_5)
logo_temp_10_cuda.upload(logo_temp_10)

# Rmark: the smalles image is probably too small for the GPU to handle (ROI error)
#kp_temp_orb_1_cuda, des_temp_orb_1_cuda = orb.detectAndComputeAsync(logo_temp_1_cuda, None)
kp_temp_orb_2_cuda, des_temp_orb_2 = orb.detectAndComputeAsync(logo_temp_2_cuda,None)
kp_temp_orb_5_cuda, des_temp_orb_5 = orb.detectAndComputeAsync(logo_temp_5_cuda,None)
kp_temp_orb_10_cuda, des_temp_orb_10 = orb.detectAndComputeAsync(logo_temp_10_cuda,None)

#kp_temp_orb_1 = kp_temp_orb_1_cuda.download()
#des_temp_orb_1 = des_temp_orb_1_cuda.download()
#kp_temp_orb_2 = kp_temp_orb_2_cuda.download()
kp_temp_orb_2 = orb.convert(kp_temp_orb_2_cuda)
des_temp_orb_2 = des_temp_orb_2.download()
#kp_temp_orb_5 = kp_temp_orb_5_cuda.download()
kp_temp_orb_5 = orb.convert(kp_temp_orb_5_cuda)
des_temp_orb_5 = des_temp_orb_5.download()
#kp_temp_orb_10 = kp_temp_orb_10_cuda.download()
kp_temp_orb_10 = orb.convert(kp_temp_orb_10_cuda)
des_temp_orb_10 = des_temp_orb_10.download()
#defining the function containing all the processing steps
def pose_est(kp_temp_orb_2, des_temp_orb_2,
             kp_temp_orb_5, des_temp_orb_5,kp_temp_orb_10, des_temp_orb_10,
             logo_temp_2_shape,logo_temp_5_shape,logo_temp_10_shape,
             K,orb,bf_HAMMING,src_gray):
    if rel_pos_c[2] < 3.0:
            #computing ORB estimates for all different resolution percentages
            zerarr = np.array([[0.0],
                    [0.0],
                    [0.0]])
            src_cuda = cv.cuda_GpuMat()
            src_gray_2 = src_gray[144:657,144:705]
            src_cuda.upload(src_gray_2)
            kp_target_cuda, des_target = orb.detectAndComputeAsync(src_cuda,None)
            kp_target = orb.convert(kp_target_cuda)
            #des_target = orb.convert(des_target_cuda)
            des_target = des_target.download()
            # des_target = des_target_cuda.download()
            if len(kp_target) == 0:
                return np.append(zerarr,np.zeros((1,1)),axis = 0)
            #https://learnopencv.com/getting-started-opencv-cuda-module/
            #https://docs.opencv.org/4.x/d5/dc3/group__cudalegacy.html
            #https://developer.nvidia.com/cuda-gpus
            #est_orb_1, inlier_am_1= det.detect_match(K,kp_temp_orb_1,des_temp_orb_1,logo_temp_1_shape,kp_target,des_target,bf_HAMMING,src_gray)
            res = 0
            if rel_pos_c[2] > 1.76:
                res = 2
                est_orb, inlier_am= det.detect_match(K,kp_temp_orb_2,des_temp_orb_2,logo_temp_2_shape,kp_target,des_target,bf_HAMMING,src_gray,logo_temp_2)
            elif rel_pos_c[2] > 0.46:
                res = 5
                est_orb, inlier_am= det.detect_match(K,kp_temp_orb_5,des_temp_orb_5,logo_temp_5_shape,kp_target,des_target,bf_HAMMING,src_gray,logo_temp_5)
            else:
                res = 10
                est_orb,inlier_am = det.detect_match(K,kp_temp_orb_10,des_temp_orb_10,logo_temp_10_shape,kp_target,des_target,bf_HAMMING,src_gray,logo_temp_10)

            if inlier_am < 20:
                trans = zerarr
            else:
                trans = est_orb
            est_orb_res = np.append(trans,np.array([[1]])*res,axis = 0)
            #computing SIFT estimate for 2 pORB_matches =cv.drawMatches(logo_temp_2, kp_temp_orb_2, src_gray, kp_target, matches[0:len(matches)], None, flags=2)
            # plt.imshow(ORB_matches),plt.show()t_match(K,kp_temp_sift_2,des_temp_sift_2,logo_temp_2.shape,sift,bf_L2,src_gray)
            # est_sift_5,inliers = det.detect_match(K,kp_temp_sift_5,des_temp_sift_5,logo_temp_5.shape,sift,bf_L2,src_gray)
            # est_sift_10,inliers = det.detect_match(K,kp_temp_sift_10,des_temp_sift_10,logo_temp_10.shape,sift,bf_L2,src_gray)
            # est_sift_lst = [est_sift_1,est_sift_2,est_sift_5,est_sift_10]
            #selecting which orb estimate is closest to the altitude estimated on drone 
            # (assuming this provides a measure of the general accuracy)
            # est_sift_res = det.resolution_sel(est_sift_lst,rel_pos_c)
            #est_orb_res = det.resolution_sel(est_orb_lst,rel_pos_c)
            # trans_est_sift.append(list(est_sift_res.astype(np.float64)))
    else:
        #try:
            src_cuda = cv.cuda_GpuMat()
            temp_size = logo_temp_2_shape
            src_gray_orig = np.copy(src_gray)
            src_gray = src_gray[144:657,144:705]
            src_cuda.upload(src_gray)
            kp_target_cuda, des_target_cuda = orb.detectAndComputeAsync(src_cuda,None)
            kp_target = orb.convert(kp_target_cuda)
            des_target = des_target_cuda.download()
            #matching keypoints 
            matches = bf_HAMMING.match(des_temp_orb_2, des_target)
            #import pdb; pdb.set_trace()
            matches = sorted(matches,key=lambda x:x.distance)
            # creating keypoint list and template corner list
            # ORB_matches =cv.drawMatches(logo_temp_2, kp_temp_orb_2, src_gray, kp_target, matches[0:len(matches)], None, flags=2)
            # plt.imshow(ORB_matches),plt.show()
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
                try: 
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
                    col = [0, 0, 0, 1]
                    # plt.plot(
                    #         A[:, 0],
                    #         A[:, 1],
                    #         "o",
                    #         markerfacecolor=tuple(col),
                    #         markeredgecolor="k",
                    #         markersize=14,
                    #         )
                    # plt.show()
                    #computing bounding box around the selected cluster
                    corn_target = dw.findBB_rot(A)
                    #putting the BB corner coordinates back to the original camera image coordinates
                    for m in range(0, len(corn_target)):
                        corn_target[m,:] = np.array([corn_target[m,0]+144,corn_target[m,1]+144])
                    #img_cont1 = np.copy(src_gray_orig)
                    # cv.drawContours(img_cont1,[corn_target],0,(255,0,0),1)
                    # plt.imshow(img_cont1),plt.show()
                    #rearranging BB points in clockwise direction
                    # https://pyimagesearch.com/2016/03/21/ordering-coordinates-clockwise-with-python-and-opencv/ 
                    ordered_pts = np.zeros((4,2))
                    s = np.sum(corn_target, axis = 1)
                    ordered_pts[0,:] = corn_target[np.argmin(s)]
                    ordered_pts[2,:] = corn_target[np.argmax(s)]
                    diff = np.diff(corn_target, axis = 1)
                    ordered_pts[1,:] = corn_target[np.argmin(diff)]
                    ordered_pts[3,:] = corn_target[np.argmax(diff)]
                    #ordered_pts = list(ordered_pts)
                    corn1_arr = np.kron(np.ones((4,1)),ordered_pts[0,:])
                    dists = np.sum(np.abs(np.subtract(ordered_pts,corn1_arr)),1)
                    closest_corn = np.where(dists == min(dists[1:4]))
                    #finding the middle positions of the close and far short sides
                    mid_close = det.mid_pos(ordered_pts[0,:],ordered_pts[closest_corn[0][0],:])
                    cp_ordered_pts = np.copy(ordered_pts)
                    cp_ordered_pts = np.delete(cp_ordered_pts,closest_corn[0][0],0)
                    cp_ordered_pts = np.delete(cp_ordered_pts,0,0)
                    mid_far = det.mid_pos(cp_ordered_pts[0,:],cp_ordered_pts[1,:])
                    #finding the center of the BB and computing the equally spaced kpts from there
                    mid_rect = det.mid_pos(mid_close,mid_far)
                    upper_next_4 = det.mid_pos(mid_far,mid_rect)
                    lower_next_4 = det.mid_pos(mid_close,mid_rect)
                    upper_next_8 = det.mid_pos(upper_next_4,mid_rect)
                    lower_next_8 = det.mid_pos(lower_next_4,mid_rect)
                    central_kp_list_target = [cv.KeyPoint(upper_next_8[0],upper_next_8[1],curstom_kp_size),
                                            cv.KeyPoint(lower_next_8[0],lower_next_8[1],curstom_kp_size),
                                            cv.KeyPoint(lower_next_4[0],lower_next_4[1],curstom_kp_size),
                                            cv.KeyPoint(upper_next_4[0],upper_next_4[1],curstom_kp_size)]
                    central_kp_2_target, central_kp_des_2_target = orb_single.compute(src_gray, central_kp_list_target)
                    # src_gray_2 = cv.drawKeypoints(img_cont1,central_kp_2_target,0,(255,0,0),cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                    # plt.imshow(src_gray_2),plt.show()
                    matches = bf_HAMMING.match(central_kp_des_2, central_kp_des_2_target)

                    #reordering the BB points in function of the positions of the matched kpts
                    if len(matches) > 0 :
                        for m in range(0,len(matches)):
                            target_pt_idx = matches[m].trainIdx
                            temp_pt_idx = matches[m].queryIdx
                            temp_coord = central_kp_2_temp[temp_pt_idx].pt
                            target_coord = central_kp_2_target[target_pt_idx].pt
                            upper_kp_arr = np.kron(np.ones((4,1)),np.array([target_coord[0],target_coord[1]]))
                            inner_dists = np.sum(np.abs(np.subtract(ordered_pts,upper_kp_arr)),1)
                            closest_corn = np.where(inner_dists == min(inner_dists))
                            inner_dists = np.delete(inner_dists,closest_corn[0][0],0)
                            closest_corn_2 = np.where(inner_dists == min(inner_dists))
                            if temp_coord[1] > logo_temp_2.shape[0]/2:
                                if closest_corn[0][0] < 2 != closest_corn_2[0][0] < 2:
                                    ordered_pts = np.roll(np.array(ordered_pts),-1,axis = 0)
                                elif closest_corn[0][0] < 2 and closest_corn_2[0][0] < 2:
                                    ordered_pts = np.roll(np.array(ordered_pts),-2,axis = 0)
                            else: 
                                if closest_corn[0][0] > 2 != closest_corn_2[0][0] > 2:
                                    ordered_pts = np.roll(np.array(ordered_pts),-1,axis = 0)
                                elif closest_corn[0][0] > 2 and closest_corn_2[0][0] > 2:
                                    ordered_pts = np.roll(np.array(ordered_pts),-2,axis = 0)
                        pos_temp_world,pos_temp_world_lst = dw.world_coord(np.array(corn_temp),logo_temp_2.shape,0)
                        dist_coeffs = np.zeros((4,1))
                        (suc,rot,trans) = cv.solvePnP(pos_temp_world, ordered_pts, K, dist_coeffs, flags=cv.SOLVEPNP_ITERATIVE)
                    else:
                        trans = np.zeros((3,1))

                    # ORB_matches_2 =cv.drawMatches(logo_temp_2, central_kp_2_temp, src_gray_orig, central_kp_2_target, matches[0:len(matches)], None, flags=2)
                    # plt.imshow(ORB_matches_2),plt.show()
                    est_orb_res = np.append(trans,np.zeros((1,1)),axis = 0)
                    # print('Chosen rotation')
                    # print(pts_arr[kp_dist_diff.index(min(kp_dist_diff))])
                except:
                    trans = np.zeros((3,1))
                    est_orb_res = np.append(trans,np.zeros((1,1)),axis = 0)
            else:
                trans = np.zeros((3,1))
                est_orb_res = np.append(trans,np.zeros((1,1)),axis = 0)

    return est_orb_res


#defining the arrays in which the results will be stored 
trans_est_orb = [] 
drone_est = []
trans_est_sift = []
tot_timing = []
#parsing the workable dataset for method evaluation
for ep in range(1,3):
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
    for observed_pos in range(0,set_size[0]):
    #for observed_pos in range(197,198):
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
        # cProfile.run("pose_est(kp_temp_orb_2, des_temp_orb_2, kp_temp_orb_5, des_temp_orb_5,kp_temp_orb_10, des_temp_orb_10,logo_temp_2.shape,logo_temp_5.shape,logo_temp_10.shape, K,orb,bf_HAMMING,src_gray)",
        #             "multi_eval_T265_threshold_step_resolution.dat")
        start = time.time()
        est_orb_res = pose_est(kp_temp_orb_2, des_temp_orb_2, kp_temp_orb_5, des_temp_orb_5,kp_temp_orb_10, des_temp_orb_10,logo_temp_2.shape,logo_temp_5.shape,logo_temp_10.shape, K,orb,bf_HAMMING,src_gray)
        end = time.time()
        timing = end - start
        tot_timing.append(timing)
        trans_est_orb.append(list(est_orb_res.astype(np.float64)))



hdf5_data = {"trans_est_orb": trans_est_orb,"drone_est":drone_est, 'timing':tot_timing}
#exporting the computations to hdf5
current_dir = '/home/gaetan/data/hdf5/'
def dump(output_dir,hdf5_data,ep):
        print('stored data in',output_dir)
        output_hdf5_path = output_dir + 'multi_eval_T265_threshold_step_resolution_cuda_match_kp_alt' + '.hdf5'
        hdf5_file = h5py.File(output_hdf5_path, "a")
        episode_group = hdf5_file.create_group(str(ep))
        for sensor_name in hdf5_data.keys():
            episode_group.create_dataset(
                sensor_name, data=np.stack(hdf5_data[sensor_name])
            )
        hdf5_file.close()
dump(current_dir,hdf5_data,'multi_eval_T265_threshold_step_resolution_cuda_match_kp_alt')



