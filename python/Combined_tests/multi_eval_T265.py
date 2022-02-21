from matplotlib import pyplot as plt
import cv2 as cv
import numpy as np
import h5py
import crop
import transform_mat as tm
import detect_match as det
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
print(base_items)
dset1 = f.get('1')
K = np.array(dset1.get('K')).reshape((3,3))

# Load other parameters and images from chosen episode
dset2 = f.get('1')
group_items = list(dset2.items())
imgs = np.array(dset2.get('observation'))
pos = np.array(dset2.get('position'))
quat = np.array(dset2.get('orientation'))
rel_pos = np.array(dset2.get('relative_position'))
img_stamp = np.array(dset2.get('image_time'))
pose_stamp = np.array(dset2.get('pose_time'))
set_size = imgs.shape

#defining the different detectors and matchers
orb = cv.ORB_create(2000,1.1,8,11,0,2,0,11,5)
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

kp_temp_sift_1, des_temp_sift_1 = sift.detectAndCompute(logo_temp_1,None)
kp_temp_sift_2, des_temp_sift_2 = sift.detectAndCompute(logo_temp_2,None)
kp_temp_sift_5, des_temp_sift_5 = sift.detectAndCompute(logo_temp_5,None)
kp_temp_sift_10, des_temp_sift_10 = sift.detectAndCompute(logo_temp_10,None)

#defining the arrays in which the results will be stored 

trans_est_orb = [] 
drone_est = []
trans_est_sift = []
#parsing the workable dataset for method evaluation
for observed_pos in range(120,set_size[0]):
    src = imgs[observed_pos,:,:,:]*255
    src_gray = np.uint8(src)

    #computing ORB estimates for all different resolution percentages
    est_orb_1,inl = det.detect_match(K,kp_temp_orb_1,des_temp_orb_1,logo_temp_1.shape,orb,bf_HAMMING,src_gray)
    est_orb_2,inl = det.detect_match(K,kp_temp_orb_2,des_temp_orb_2,logo_temp_2.shape,orb,bf_HAMMING,src_gray)
    est_orb_5,inl = det.detect_match(K,kp_temp_orb_5,des_temp_orb_5,logo_temp_5.shape,orb,bf_HAMMING,src_gray)
    est_orb_10,inl = det.detect_match(K,kp_temp_orb_10,des_temp_orb_10,logo_temp_10.shape,orb,bf_HAMMING,src_gray)
    est_orb_lst = [est_orb_1,est_orb_2,est_orb_5,est_orb_10]
    #computing SIFT estimate for 2 percent resolution (for scale invariance performance test)
    #est_sift_1,inliers = det.detect_match(K,kp_temp_sift_1,des_temp_sift_1,logo_temp_1.shape,sift,bf_L2,src_gray)
    est_sift_2,inliers = det.detect_match(K,kp_temp_sift_2,des_temp_sift_2,logo_temp_2.shape,sift,bf_L2,src_gray)
    # est_sift_5,inliers = det.detect_match(K,kp_temp_sift_5,des_temp_sift_5,logo_temp_5.shape,sift,bf_L2,src_gray)
    # est_sift_10,inliers = det.detect_match(K,kp_temp_sift_10,des_temp_sift_10,logo_temp_10.shape,sift,bf_L2,src_gray)
    # est_sift_lst = [est_sift_1,est_sift_2,est_sift_5,est_sift_10]
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
    drone_est.append(list(rel_pos_c.astype(np.float64)))
    #selecting which orb estimate is closest to the altitude estimated on drone 
    # (assuming this provides a measure of the general accuracy)
    #est_sift_res = det.resolution_sel(est_sift_lst,rel_pos_c)
    est_orb_res = det.resolution_sel(est_orb_lst,rel_pos_c)
    trans_est_sift.append(list(est_sift_2.astype(np.float64)))
    trans_est_orb.append(list(est_orb_res.astype(np.float64)))

hdf5_data = {"trans_est_orb": trans_est_orb,"drone_est":drone_est,"trans_est_sift":trans_est_sift}
#exporting the computations to hdf5
current_dir = '/home/gaetan/data/hdf5/'
def dump(output_dir,hdf5_data,ep):
        print('stored data in',output_dir)
        output_hdf5_path = output_dir + '/multi_eval_data_15' + '.hdf5'
        hdf5_file = h5py.File(output_hdf5_path, "a")
        episode_group = hdf5_file.create_group(str(ep))
        for sensor_name in hdf5_data.keys():
            episode_group.create_dataset(
                sensor_name, data=np.stack(hdf5_data[sensor_name])
            )
        hdf5_file.close()
dump(current_dir,hdf5_data,'multi_eval_data_15')
