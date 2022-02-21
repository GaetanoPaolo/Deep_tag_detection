from matplotlib import pyplot as plt
import cv2 as cv
import numpy as np
import h5py
import crop
import itertools
import draw_transf as dw
import transform_mat

#load the logo template
logo_temp = cv.imread('/home/gaetan/code/simulation_ws/src/my_simulations/models/psi_logo/materials/textures/poster-psi-drone-logo-2percent.png',0)
rot = 0
scale = 2
logo_temp = crop.crop_img(logo_temp,scale)

# load the labelled gazebo data
f = h5py.File('/home/gaetan/data/hdf5/correct_baselink_gt/data4_correct_gt.hdf5', 'r+')
base_items = list(f.items())
dset = f.get('labelled_data')
imgs = np.array(dset.get('observation'))
corn = np.array(dset.get('corners'))
pos_origin_cam = np.array(dset.get('pos_origin_cam'))
pos = np.array(dset.get('position'))
quat = np.array(dset.get('orientation'))
pos_down_link = np.array(dset.get("pos_down_link"))
quat_down_link = np.array(dset.get("quat_down_link"))
pos_down_optical_frame = np.array(dset.get("pos_down_optical_frame"))
quat_down_optical_frame = np.array(dset.get("quat_down_optical_frame"))
hdf5_data = {"gt_cam_pos": [], "est_cam_pos": []}

#arrays for output storage: estimated positions in trans_est 
# and corresponding GT in pos_origin_list
trans_est = []
pos_origin_list = []

#initiating necessary parameters
fx = 119
fy = 119
arr_size = pos_origin_cam.shape
pos_origin_cam = np.transpose(pos_origin_cam)[0,0:3,:]
img_size = imgs[0,:,:,:].shape
K = np.array([[fx,0,img_size[1]/2],
                [0,fy,img_size[0]/2],
                [0,0,1]])

#code source till line 31: https://datahacker.rs/feature-matching-methods-comparison-in-opencv/
#creating keypoint matchers and finders
orb = cv.ORB_create(500,1.1,8,21,0,2,0,21,20)
bf = cv.BFMatcher_create(cv.NORM_HAMMING,crossCheck=True)
#calculating template keypoints and descriptors before the loop
kp_temp, des_temp = orb.detectAndCompute(logo_temp,None)
train_dim = imgs.shape

trans_corn = np.zeros((3,arr_size[0]))
arr_world_cam_est = np.zeros((3,arr_size[0]))
count = 0
T_baselink_downlink = transform_mat.transf_mat(quat_down_link[0,:],pos_down_link[0,:])
T_downlink_downoptframe = transform_mat.transf_mat(quat_down_optical_frame[0,:],pos_down_optical_frame[0,:])
print("Amount of usable sample image")
print(train_dim[0]-200)
for k in range(200,train_dim[0]):
    src = imgs[k,:,:,:]*255
    src_gray = np.uint8(cv.cvtColor(src, cv.COLOR_BGR2GRAY))
    kp_target, des_target = orb.detectAndCompute(src_gray,None)
    if type(des_target) != type(None):
        T_world_baselink = transform_mat.transf_mat(quat[k,:],pos[k,:])
        T_world_downlink = np.matmul(T_world_baselink,T_baselink_downlink)
        T_world_downoptframe = np.matmul(T_world_downlink,T_downlink_downoptframe)
        hdf5_data["gt_cam_pos"].append(np.squeeze(T_world_downoptframe[0:3,3],axis = 1))
        matches = bf.match(des_temp, des_target)
        matches = sorted(matches,key=lambda x:x.distance)
        pos_temp = []
        pos_target = []
        for m in range(0,len(matches)):
            target_pt_idx = matches[m].trainIdx
            temp_pt_idx = matches[m].queryIdx
            temp_coord = kp_temp[temp_pt_idx].pt
            target_coord = kp_target[target_pt_idx].pt
            pos_temp.append(temp_coord)
            pos_target.append(target_coord)
        pos_temp_hom = np.float32(pos_temp).reshape(-1,1,2)
        pos_target_hom = np.float32(pos_target).reshape(-1,1,2)
        if len(matches) > 20:
            pos_temp_world = dw.world_coord(np.array(pos_temp),logo_temp,rot)
            dist_coeffs = np.zeros((4,1))
            (suc,est_rot,est_trans,inliers) = cv.solvePnPRansac(pos_temp_world, np.array(pos_target), K, dist_coeffs, flags=cv.SOLVEPNP_ITERATIVE, iterationsCount=2000, reprojectionError=3.0)
            rot_out = cv.Rodrigues(est_rot)
            rot_mat = rot_out[0]
            T_cam_world = np.matrix([[rot_mat[0,0], rot_mat[0,1], rot_mat[0,2], est_trans[0,0]],
                      [rot_mat[1,0], rot_mat[1,1], rot_mat[1,2], est_trans[1,0]],
                      [rot_mat[2,0], rot_mat[2,1], rot_mat[2,2], est_trans[2,0]],
                      [0,0,0,1]])
            T_world_cam = np.linalg.inv(T_cam_world)
            p_cam_origin = np.array([[0],
                    [0],
                    [0],
                    [1]])
            p_world_cam_est = np.matmul(T_world_cam,p_cam_origin)
            arr_world_cam_est[:,k] = np.squeeze(p_world_cam_est[0:3,0],axis = 1)
            hdf5_data["est_cam_pos"].append(np.squeeze(p_world_cam_est[0:3,0],axis = 1))
            trans_est.append(est_trans[:,0])
            pos_origin_list.append(pos_origin_cam[0:3,k])


diff = np.subtract(np.array(trans_est),np.array(pos_origin_list))
print('Amount of samples with more than 20 matches')
print(diff.shape[0])
mse = np.mean(np.square(diff), axis = 0)
print("MSE for each dimension of the relative position estimation (in cm^2)")
print(mse)


#storing world coords in hdf5 (optional)
# current_dir = '/home/gaetan/data/hdf5/correct_baselink_gt/'
# for i in range(1,len(hdf5_data["est_cam_pos"])):
#     print(hdf5_data["est_cam_pos"][i].shape),
#     print(hdf5_data["gt_cam_pos"][i].shape)

def dump(output_dir,hdf5_data,ep):
        print('stored data in',output_dir)
        output_hdf5_path = output_dir + '/3D_pos' + '.hdf5'
        hdf5_file = h5py.File(output_hdf5_path, "a")
        episode_group = hdf5_file.create_group(str(ep))
        for sensor_name in hdf5_data.keys():
            episode_group.create_dataset(
                sensor_name, data=np.stack(hdf5_data[sensor_name])
            )
        hdf5_file.close()
#dump(current_dir,hdf5_data,'3D_pos')



