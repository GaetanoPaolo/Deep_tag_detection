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
trans_est_orb =[]
trans_est_sift = []
drone_est = []
#parsing the workable dataset for method evaluation
for observed_pos in range(120,set_size[0]):
    src = imgs[observed_pos,:,:,:]*255
    src_gray = np.uint8(src)

    #computing ORB estimates for all different resolution percentages
    est_orb_1 = det.detect_match(K,kp_temp_orb_1,des_temp_orb_1,logo_temp_1.shape,orb,bf_HAMMING,src_gray)
    est_orb_2 = det.detect_match(K,kp_temp_orb_2,des_temp_orb_2,logo_temp_2.shape,orb,bf_HAMMING,src_gray)
    est_orb_5 = det.detect_match(K,kp_temp_orb_5,des_temp_orb_5,logo_temp_5.shape,orb,bf_HAMMING,src_gray)
    est_orb_10 = det.detect_match(K,kp_temp_orb_10,des_temp_orb_10,logo_temp_10.shape,orb,bf_HAMMING,src_gray)
    est_orb_lst = [est_orb_1,est_orb_2,est_orb_5,est_orb_10]

    #computing SIFT estimate for 2 percent resolution (for scale invariance performance test)
    est_sift_2 = det.detect_match(K,kp_temp_sift_2,des_temp_sift_2,logo_temp_2.shape,sift,bf_L2,src_gray)
    trans_est_sift.append(est_sift_2[:,0])
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
    drone_est.append(rel_pos_c[:,0])
    #selecting which orb estimate is closest to the altitude estimated on drone 
    # (assuming this provides a measure of the general accuracy)
    alt_diff = []
    for i in range(0,len(est_orb_lst)):
        cur_est = est_orb_lst[i]
        cur_diff = abs(rel_pos_c[2,0]-cur_est[2,0])
        alt_diff.append(cur_diff)
    min_alt_diff = min(alt_diff)
    min_index = alt_diff. index(min_alt_diff) 
    
    #extending the orb estimate with the best resolution
    res_list = [1,2,5,10]
    ext_orb_res = np.zeros((4,1))
    ext_orb_res[0:3,0] = np.squeeze(est_orb_lst[min_index],axis = 1)
    ext_orb_res[3,0] = res_list[min_index]
    trans_est_orb.append(ext_orb_res)

trans_est_orb =np.array(trans_est_orb)
trans_est_sift = np.array(trans_est_sift)
drone_est = np.array(drone_est)

#computing errors
orb_err = np.squeeze(abs(np.subtract(drone_est,trans_est_orb[:,0:3,:])),axis = 2)
sift_err = np.squeeze(abs(np.subtract(drone_est,trans_est_sift)),axis = 2)
drone_est = np.squeeze(drone_est, axis = 2)
#plot each axis along altitude
fig, axd = plt.subplot_mosaic([['fist'],
                               ['second'],
                               ['third'],
                               ['fourth']], layout='constrained')
axd['first'].set_title('ORB best resolution percentage')
axd['first'].plot(drone_est[:,2],trans_est_orb[:,3])
axd['second'].set_title('Errors x-axis')
axd['second'].plot(drone_est[:,2],orb_err[:,0], label = 'ORB')
axd['second'].plot(drone_est[:,2],sift_err[:,0], label = 'SIFT')
axd['second'].legend()
axd['third'].set_title('Errors y-axis')
axd['third'].plot(drone_est[:,2],orb_err[:,1], label = 'ORB')
axd['third'].plot(drone_est[:,2],sift_err[:,1], label = 'SIFT')
axd['third'].legend()
axd['fourth'].set_title('Errors z-axis')
axd['fourth'].plot(drone_est[:,2],orb_err[:,2], label = 'ORB')
axd['fourth'].plot(drone_est[:,2],sift_err[:,2], label = 'SIFT')
axd['fourth'].legend()

