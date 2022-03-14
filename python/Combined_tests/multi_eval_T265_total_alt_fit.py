from matplotlib import pyplot as plt
import cv2 as cv
import numpy as np
import h5py
import crop
import transform_mat as tm
import detect_match as det
import draw_transf as dw
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
        if rel_pos_c[2] < 3.0:
            #computing ORB estimates for all different resolution percentages
            est_orb_1,inl = det.detect_match(K,kp_temp_orb_1,des_temp_orb_1,logo_temp_1.shape,orb,bf_HAMMING,src_gray)
            est_orb_2,inl = det.detect_match(K,kp_temp_orb_2,des_temp_orb_2,logo_temp_2.shape,orb,bf_HAMMING,src_gray)
            est_orb_5,inl = det.detect_match(K,kp_temp_orb_5,des_temp_orb_5,logo_temp_5.shape,orb,bf_HAMMING,src_gray)
            est_orb_10,inl = det.detect_match(K,kp_temp_orb_10,des_temp_orb_10,logo_temp_10.shape,orb,bf_HAMMING,src_gray)
            est_orb_lst = [est_orb_1,est_orb_2,est_orb_5,est_orb_10]
            #computing SIFT estimate for 2 percent resolution (for scale invariance performance test)
            # est_sift_1,inliers = det.detect_match(K,kp_temp_sift_1,des_temp_sift_1,logo_temp_1.shape,sift,bf_L2,src_gray)
            # est_sift_2,inliers = det.detect_match(K,kp_temp_sift_2,des_temp_sift_2,logo_temp_2.shape,sift,bf_L2,src_gray)
            # est_sift_5,inliers = det.detect_match(K,kp_temp_sift_5,des_temp_sift_5,logo_temp_5.shape,sift,bf_L2,src_gray)
            # est_sift_10,inliers = det.detect_match(K,kp_temp_sift_10,des_temp_sift_10,logo_temp_10.shape,sift,bf_L2,src_gray)
            # est_sift_lst = [est_sift_1,est_sift_2,est_sift_5,est_sift_10]
            #selecting which orb estimate is closest to the altitude estimated on drone 
            # (assuming this provides a measure of the general accuracy)
            # est_sift_res = det.resolution_sel(est_sift_lst,rel_pos_c)
            est_orb_res = det.resolution_sel(est_orb_lst,rel_pos_c)
            # trans_est_sift.append(list(est_sift_res.astype(np.float64)))
        else:
            try:
                temp_size = logo_temp_2.shape
                src_gray = src_gray[144:657,144:705]
                kp_target, des_target = orb.detectAndCompute(src_gray,None)
                # creating keypoint list and template corner list
                corn_temp = [[0,0],[temp_size[1],0],[temp_size[1],temp_size[0]],[0,temp_size[0]]]
                #finding the corner points
                #finding the corner points
                diag_temp = np.linalg.norm(np.array([temp_size[0],temp_size[1]]))
                temp_ratio = temp_size[1]/diag_temp
                Z = det.kp_preproc(kp_target)
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
                        if  ratio_diff < 0.3:
                            stop = True
                            break
                    count += 1
                # print('Corn target')
                # print(corn_target)
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
                point_amount = 4*rot_step
                ordered_pts = det.rectangle_side_fractal(ordered_pts,point_amount)
                corn_temp = det.rectangle_side_fractal(corn_temp,point_amount)

                #checking the mean keypoint distances relative to the centers of the long sides
                #long sides : 6, 14
                # short sides : 2, 10
                # for varying values of n: 
                #long sides: rot_step + 2**(n-1), 3*rot_step + 2**(n-1)
                #short sides: 2**(n-1), 2*rot_step + 2**(n-1)
                pt1 = corn_temp[6]
                pt2 = corn_temp[14]
                dst1 = det.mean_keypoint_dist_temp(kp_temp_orb_2,pt1)
                dst2 = det.mean_keypoint_dist_temp(kp_temp_orb_2,pt2)
                dst_temp = np.array([dst1,dst2])
                # print('dst_temp')
                # print(dst_temp)
                rot = 0
                pos_temp =  corn_temp
                dist_coeffs = np.array([-0.014216,0.060412,-0.054711,0.011151])
                candidate_trans = []
                inlier_arr = [0]
                pts_arr = []
                for shift in range(0,4):
                    ordered_pts_2 = list(np.roll(np.array(ordered_pts),-rot_step*shift,axis = 0))
                    pos_temp_world = dw.world_coord(np.array(pos_temp),logo_temp_2.shape,rot)
                    pos_target = ordered_pts_2
                    (suc,angle,trans,inliers) = cv.solvePnPRansac(pos_temp_world, np.array(pos_target), K, dist_coeffs, flags=cv.SOLVEPNP_ITERATIVE, iterationsCount=2000, reprojectionError=8.0)
                    if inliers is not None:
                        if len(inliers) >= max(inlier_arr):
                            candidate_trans.append(trans)
                            pts_arr.append(ordered_pts_2)
                        inlier_arr.append(len(inliers))
                kp_dist_diff = []
                for k in range(0,len(pts_arr)):
                    pt1 = pts_arr[k][6]
                    pt2 = pts_arr[k][14]
                    dst1 = det.mean_keypoint_dist_target(A,pt1)
                    dst2 = det.mean_keypoint_dist_target(A,pt2)
                    dst_target = np.array([dst1,dst2])
                    diff = np.linalg.norm(np.subtract(dst_temp,dst_target))
                    kp_dist_diff.append(diff)
                trans = candidate_trans[kp_dist_diff.index(min(kp_dist_diff))]
                if np.sum(trans) > 100:
                    trans = np.zeros((3,1))
                # print('Chosen rotation')
                # print(pts_arr[kp_dist_diff.index(min(kp_dist_diff))])

            except:
                trans = np.zeros((3,1))
            est_orb_res = np.append(trans,np.zeros((1,1)),axis = 0)
        trans_est_orb.append(list(est_orb_res.astype(np.float64)))

hdf5_data = {"trans_est_orb": trans_est_orb,"drone_est":drone_est}
#exporting the computations to hdf5
current_dir = '/home/gaetan/data/hdf5/'
def dump(output_dir,hdf5_data,ep):
        print('stored data in',output_dir)
        output_hdf5_path = output_dir + '/T265_alt_fit' + '.hdf5'
        hdf5_file = h5py.File(output_hdf5_path, "a")
        episode_group = hdf5_file.create_group(str(ep))
        for sensor_name in hdf5_data.keys():
            episode_group.create_dataset(
                sensor_name, data=np.stack(hdf5_data[sensor_name])
            )
        hdf5_file.close()
dump(current_dir,hdf5_data,'T265_alt_fit')

