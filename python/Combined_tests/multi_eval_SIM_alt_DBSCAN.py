from matplotlib import pyplot as plt
import cv2 as cv
import numpy as np
import h5py
import crop
import transform_mat as tm
import detect_match_SIM as det
import draw_transf as dw
from sklearn.cluster import DBSCAN
#load the logo templates corresponding to different percentages of kept
#resolution from the original logo
logo_temp_1 = cv.imread('/home/gaetan/code/simulation_ws/src/my_simulations/models/psi_logo/materials/textures/poster-psi-drone-logo-1percent.png',0)
#logo_temp_1 = crop.crop_img(logo_temp_1,1)
logo_temp_2 = cv.imread('/home/gaetan/code/simulation_ws/src/my_simulations/models/psi_logo/materials/textures/poster-psi-drone-logo-2percent.png',0)
#logo_temp_2 = crop.crop_img(logo_temp_2,2)
logo_temp_5 = cv.imread('/home/gaetan/code/simulation_ws/src/my_simulations/models/psi_logo/materials/textures/poster-psi-drone-logo-5percent.png',0)
#logo_temp_5 = crop.crop_img(logo_temp_5,5)
logo_temp_10 = cv.imread('/home/gaetan/code/simulation_ws/src/my_simulations/models/psi_logo/materials/textures/poster-psi-drone-logo-10percent.png',0)
#logo_temp_10 = crop.crop_img(logo_temp_10,10)
logo_temp_100 = cv.imread('/home/gaetan/code/simulation_ws/src/my_simulations/models/psi_logo/materials/textures/poster-psi-drone-logo.png',0)

# load the camera parameters stored in episode 1
f = h5py.File('/home/gaetan/data/hdf5/psi_800res_alt_rot_qr/data4_sync.hdf5', 'r+')
base_items = list(f.items())
print(base_items)
dset2 = f.get('1')
imgs = np.array(dset2.get('observation'))
#Defining the correct intrinsic parameter matrix
img_size = imgs[0,:,:,:].shape
horizontal_field_of_view = (80 * img_size[1]/img_size[0]) * 3.14 / 180
vertical_field_of_view = 80 * 3.14 / 180
fx = img_size[1]/2*np.tan(horizontal_field_of_view/2)**(-1)
fy = img_size[0]/2*np.tan(vertical_field_of_view/2)**(-1)

K = np.array([[fx,0,img_size[1]/2],
                [0,fy,img_size[0]/2],
                [0,0,1]])

#defining the different detectors and matchers
#orb = cv.ORB_create(2000,1.1,8,11,0,2,0,11,5)
orb = cv.ORB_create(2000,1.1,8,21,0,2,0,21,20)
#sift = cv.SIFT_create(2000,6,0.02,30,1.6,cv.CV_32F)
sift = cv.SIFT_create(2000,3,0.02,30,1.4,cv.CV_32F)
brisk = cv.BRISK_create(3,6,0.9)
bf_HAMMING = cv.BFMatcher_create(cv.NORM_HAMMING,crossCheck=True)
bf_L2= cv.BFMatcher_create(cv.NORM_L2,crossCheck=True)

#a priori detection of keypoints on the templates using the different methods
kp_temp_orb_1, des_temp_orb_1 = orb.detectAndCompute(logo_temp_1,None)
kp_temp_orb_2, des_temp_orb_2 = orb.detectAndCompute(logo_temp_2,None)
kp_temp_orb_5, des_temp_orb_5 = orb.detectAndCompute(logo_temp_5,None)
kp_temp_orb_10, des_temp_orb_10 = orb.detectAndCompute(logo_temp_10,None)
kp_temp_orb_100, des_temp_orb_100 = orb.detectAndCompute(logo_temp_100,None)
# kp_temp_brisk_1, des_temp_brisk_1 = brisk.detectAndCompute(logo_temp_1,None)
# kp_temp_brisk_2, des_temp_brisk_2 = brisk.detectAndCompute(logo_temp_2,None)
# kp_temp_brisk_5, des_temp_brisk_5 = brisk.detectAndCompute(logo_temp_5,None)
# kp_temp_brisk_10, des_temp_brisk_10 = brisk.detectAndCompute(logo_temp_10,None)

# kp_temp_sift_1, des_temp_sift_1 = sift.detectAndCompute(logo_temp_1,None)
# kp_temp_sift_2, des_temp_sift_2 = sift.detectAndCompute(logo_temp_2,None)
# kp_temp_sift_5, des_temp_sift_5 = sift.detectAndCompute(logo_temp_5,None)
# kp_temp_sift_10, des_temp_sift_10 = sift.detectAndCompute(logo_temp_10,None)

#defining the arrays in which the results will be stored 

trans_est_orb = [] 
drone_est = []
trans_est_sift = []
prev_alt = 0
#parsing the workable dataset for method evaluation
for ep in range(1,len(base_items)+1):
#for ep in range(15,15+1):
    # Load other parameters and images from chosen episode
    print(ep)
    ep_str = "% s" % ep
    dset2 = f.get(ep_str)
    group_items = list(dset2.items())
    pos_origin_cam = np.array(dset2.get('pos_origin_cam'))
    arr_size = pos_origin_cam.shape
    pos_origin_cam = np.transpose(pos_origin_cam)[0,0:3,:]
    imgs = np.array(dset2.get('observation'))
    pos = np.array(dset2.get('position'))
    quat = np.array(dset2.get('orientation'))
    #rel_pos = np.array(dset2.get('relative_position'))
    img_stamp = np.array(dset2.get('image_time'))
    pose_stamp = np.array(dset2.get('pose_time'))
    set_size = imgs.shape
    for observed_pos in range(0,set_size[0]):
    #for observed_pos in range(0,1):
        print(observed_pos)
        src = imgs[observed_pos,:,:,:]*255
        src_gray = np.uint8(cv.cvtColor(src, cv.COLOR_BGR2GRAY))
        rel_pos_c = pos_origin_cam[:,observed_pos]
        drone_est.append(list(rel_pos_c.astype(np.float64)))
        #computing ORB estimates for all different resolution percentages
        if rel_pos_c[2] < 3.0:
            inliers1,est_orb_1 = det.detect_match(K,kp_temp_orb_1,des_temp_orb_1,logo_temp_1.shape,orb,bf_HAMMING,src_gray)
            inliers2,est_orb_2 = det.detect_match(K,kp_temp_orb_2,des_temp_orb_2,logo_temp_2.shape,orb,bf_HAMMING,src_gray)
            inliers5,est_orb_5 = det.detect_match(K,kp_temp_orb_5,des_temp_orb_5,logo_temp_5.shape,orb,bf_HAMMING,src_gray)
            inliers10,est_orb_10 = det.detect_match(K,kp_temp_orb_10,des_temp_orb_10,logo_temp_10.shape,orb,bf_HAMMING,src_gray)
            inliers100,est_orb_100 = det.detect_match(K,kp_temp_orb_100,des_temp_orb_100,logo_temp_100.shape,orb,bf_HAMMING,src_gray)
            est_orb_lst = [est_orb_1,est_orb_2,est_orb_5,est_orb_10,est_orb_100]
            #inlier_lst = [inliers1,inliers2,inliers5,inliers10,inliers100]
            #computing SIFT estimate for 2 percent resolution (for scale invariance performance test)
            # est_sift_1 = det.detect_match(K,kp_temp_sift_1,des_temp_sift_1,logo_temp_1.shape,sift,bf_L2,src_gray)
            # est_sift_2 = det.detect_match(K,kp_temp_sift_2,des_temp_sift_2,logo_temp_2.shape,sift,bf_L2,src_gray)
            # est_sift_5 = det.detect_match(K,kp_temp_sift_5,des_temp_sift_5,logo_temp_5.shape,sift,bf_L2,src_gray)
            # est_sift_10 = det.detect_match(K,kp_temp_sift_10,des_temp_sift_10,logo_temp_10.shape,sift,bf_L2,src_gray)
            # est_sift_lst = [est_sift_1,est_sift_2,est_sift_5,est_sift_10]
            #computing the estimated tag position relative to the camera (from drone world axes position estimate)
            #selecting which orb estimate is closest to the altitude estimated on drone 
            #(assuming this provides a measure of the general accuracy)
            # est_sift_res = det.resolution_sel(est_sift_lst,rel_pos_c)
            est_orb_res = det.resolution_sel(est_orb_lst,rel_pos_c)
            #est_orb_res = det.resolution_sel_inliers(est_orb_lst,inlier_lst)
        else:
            temp_size = logo_temp_2.shape
            kp_target, des_target = orb.detectAndCompute(src_gray,None)
            bf = cv.BFMatcher_create(cv.NORM_HAMMING,crossCheck=True)
            #matching keypoints 
            matches = bf.match(des_temp_orb_2, des_target)
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
            pos_temp_world = dw.world_coord(np.array(pos_temp),logo_temp_2.shape,rot)
            dist_coeffs = np.zeros((4,1))
            (suc,angle,trans,inliers) = cv.solvePnPRansac(pos_temp_world, np.array(pos_target), K, dist_coeffs, flags=cv.SOLVEPNP_ITERATIVE, iterationsCount=2000, reprojectionError=2.0)
            if isinstance(inliers,type(None)):
                print(inliers)
            else:
                print(len(inliers))
                inlier_matches = []
                inlier_kp_target = []
                for k in range(0,len(inliers)):
                    inlier_matches.append(matches[inliers[k][0]])
                    cur_inlier_idx = matches[inliers[k][0]].trainIdx
                    cur_inlier_kp = kp_target[cur_inlier_idx].pt
                    int_cur_inlier_kp = [int(cur_inlier_kp[0]),int(cur_inlier_kp[1])]
                    inlier_kp_target.append(int_cur_inlier_kp)
            
            inlier_kp_target = pos_target
            #if less than 20 and at least two inliers: activation of the clustering algorithm to add edge matches
            #if not isinstance(inliers,type(None)):
            if len(matches) >= 2 and pos_origin_cam[2,observed_pos] > 3:
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
                # Number of clusters in labels, ignoring noise if present.
                n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
                n_noise_ = list(labels).count(-1)

                # #############################################################################

                # Selecting cluster as the one that contains the larges amount of matches
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
                # find which class contains the matches
                class_pos = match_counters.index(max(match_counters))
                class_member_mask = labels==class_pos
                A = Z[class_member_mask  & core_samples_mask]
                #computing bounding box around the selected cluster
                corn_target = dw.findBB_rot(A)

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

                pos_temp =  corn_temp + pos_temp
                #pos_temp =  corn_temp
                dist_coeffs = np.zeros((4,1))
                candidate_trans = []
                inlier_arr = [0]
                pts_arr = []
                for shift in range(0,4):
                    ordered_pts_2 = list(np.roll(np.array(ordered_pts),-rot_step*shift,axis = 0))
                    pos_temp_world = dw.world_coord(np.array(pos_temp),logo_temp_2.shape,rot)
                    pos_target_2 = ordered_pts_2 + pos_target
                    #pos_target_2 = ordered_pts_2
                    (suc,angle,trans,inliers) = cv.solvePnPRansac(pos_temp_world, np.array(pos_target_2), K, dist_coeffs, flags=cv.SOLVEPNP_ITERATIVE, iterationsCount=2000, reprojectionError=2.0)
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
                print(trans)
                est_orb_res = np.append(trans,np.zeros((1,1)),axis = 0)
                # except:
                #     trans = np.zeros((3,1))
                #     est_orb_res = np.append(trans,np.zeros((1,1)),axis = 0)
                #     print("zero res exception")
            else:
                trans = np.zeros((3,1))
                est_orb_res = np.append(trans,np.zeros((1,1)),axis = 0)
                print("zero res matches")
                
        #prev_alt = est_orb_res[2]
        #prev_alt = est_orb_res[2]
        trans_est_orb.append(list(est_orb_res.astype(np.float64)))
        

hdf5_data = {"trans_est_orb": trans_est_orb,"drone_est":drone_est}
#exporting the computations to hdf5
current_dir = '/home/gaetan/data/hdf5/'
def dump(output_dir,hdf5_data,ep):
        print('stored data in',output_dir)
        output_hdf5_path = output_dir + '/psi_800res_alt_rot_100_DBSCAN_epsfunc_qr' + '.hdf5'
        hdf5_file = h5py.File(output_hdf5_path, "a")
        episode_group = hdf5_file.create_group(str(ep))
        for sensor_name in hdf5_data.keys():
            print(sensor_name)
            episode_group.create_dataset(
                sensor_name, data=np.stack(hdf5_data[sensor_name])
            )
        hdf5_file.close()
dump(current_dir,hdf5_data,'psi_800res_alt_rot_100_DBSCAN_epsfunc_qr')