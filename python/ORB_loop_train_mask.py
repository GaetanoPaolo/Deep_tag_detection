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
#currently only using vertical version as rotations don't seem to improve the matching
#removing the blue edge of the logo template to detect logo itself
logo_temp = crop.crop_img(logo_temp)
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
hdf5_data = {"gt_cam_pos": [], "est_cam_pos": [],'gt_orig_pos':[],'est_orig_pos':[],'observation':[]}
#initiating necessary parameters
fx = 119
fy = 119
arr_size = pos_origin_cam.shape
img_size = imgs[0,:,:,:].shape
K = np.array([[fx,0,img_size[1]/2],
                [0,fy,img_size[0]/2],
                [0,0,1]])
#code source till line 31: https://datahacker.rs/feature-matching-methods-comparison-in-opencv/
#creating keypoint matchers and finders
orb = cv.ORB_create()
bf = cv.BFMatcher_create(cv.NORM_HAMMING,crossCheck=True)
#calculating template keypoints and descriptors before the loop
kp_temp, des_temp = orb.detectAndCompute(logo_temp,None)
train_dim = imgs.shape
trans_est = np.zeros((3,arr_size[0]))
trans_corn = np.zeros((3,arr_size[0]))
valid_est = np.zeros((1,arr_size[0]))
#arr_world_cam = np.zeros((3,arr_size[0]))
#arr_world_cam_est = np.zeros((3,arr_size[0]))
pos_origin_cam = np.transpose(pos_origin_cam)[0,0:3,:]
start = 0
count = 0
T_baselink_downlink = transform_mat.transf_mat(quat_down_link[0,:],pos_down_link[0,:])
T_downlink_downoptframe = transform_mat.transf_mat(quat_down_optical_frame[0,:],pos_down_optical_frame[0,:])
for k in range(100,train_dim[0]):
    src = imgs[k,:,:,:]*255
    src_gray = np.uint8(cv.cvtColor(src, cv.COLOR_BGR2GRAY))
    cur_corn = corn[k,:,:]
    corn_size = cur_corn.shape
    img_size = src_gray.shape
    keypts = []
    for j in range(0,corn_size[1]):
        keypts.append((int(cur_corn[0,j]),int(cur_corn[1,j])))
    mask = np.zeros(img_size[:2], dtype="uint8")
    cv.rectangle(mask, keypts[1], keypts[3], 255, -1)
    kp_target, des_target = orb.detectAndCompute(src_gray,mask)
    if type(des_target) != type(None):
        T_world_baselink = transform_mat.transf_mat(quat[k,:],pos[k,:])
        T_world_downlink = np.matmul(T_world_baselink,T_baselink_downlink)
        T_world_downoptframe = np.matmul(T_world_downlink,T_downlink_downoptframe)
        #pos_world_cam = T_world_downoptframe[0:3,3]
        #arr_world_cam[:,k] = np.squeeze(T_world_downoptframe[0:3,3],axis = 1)
        if start == 0:
            start = k
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
        M, mask = cv.findHomography( pos_temp_hom, pos_target_hom,cv.RANSAC,2.0)
        matchesMask = mask.ravel().tolist()
        inlier_matches = []
        rem = 0
        for n in range(0,len(matches)):
            if matchesMask[n] == 1:
                inlier_matches.append(matches[n])
            else:
                pos_temp.pop(n-rem)
                pos_target.pop(n-rem)
                rem += 1
        if len(inlier_matches) > 3:
            pos_temp_world = dw.world_coord(np.array(pos_temp),logo_temp,rot)
            dist_coeffs = np.zeros((4,1))
            (suc,est_rot,est_trans) = cv.solvePnP(pos_temp_world, np.array(pos_target), K, dist_coeffs, flags=cv.SOLVEPNP_ITERATIVE)
            if np.abs(est_trans[0,0]) < 1000:
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
                #arr_world_cam_est[:,k] = np.squeeze(p_world_cam_est[0:3,0],axis = 1)
                hdf5_data["est_cam_pos"].append(np.squeeze(p_world_cam_est[0:3,0],axis = 1))
                hdf5_data["gt_cam_pos"].append(np.squeeze(T_world_downoptframe[0:3,3],axis = 1))
                hdf5_data["est_orig_pos"].append(est_trans[0:3,0])
                hdf5_data["gt_orig_pos"].append(pos_origin_cam[0:3,k])
                hdf5_data["observation"].append(imgs[k,:,:,:])
                trans_est[:,k] = est_trans[:,0]
                valid_est[0,k] = 1
            else:
                trans_est[:,k] = trans_est[:,k-1]
                #hdf5_data["est_cam_pos"].append(hdf5_data["est_cam_pos"][len(hdf5_data["est_cam_pos"])-1])
        else:
            trans_est[:,k] = trans_est[:,k-1]
            #arr_world_cam_est[:,k] = arr_world_cam_est[:,k-1]
            #hdf5_data["est_cam_pos"].append(hdf5_data["est_cam_pos"][len(hdf5_data["est_cam_pos"])-1])
            #print(k)

        # corn_tup = []
        # for i in range(0,corn_size[1]):
        #     corn_tup.append((float(cur_corn[0,i]),float(cur_corn[1,i])))
            
        # abs_y_max = 0.68/2
        # abs_x_max = 0.98/2
        # z = 0.001
        # c_world = [(abs_x_max,-abs_y_max,z),(-abs_x_max,-abs_y_max,z),(-abs_x_max,abs_y_max,z),(abs_x_max,abs_y_max,z)]
        # (suc_corn,rot_corn,corn_trans) = cv.solvePnP(np.array(c_world), np.array(corn_tup), K, dist_coeffs, flags=cv.SOLVEPNP_ITERATIVE)
        # trans_corn[:,k] = corn_trans[:,0]
        #print(trans_corn[:,k])
    # print(k),
    # print(trans_est[:,k])


#pos = np.transpose(pos)
diff = np.subtract(trans_est,pos_origin_cam)
valid_diff  = np.multiply(diff,valid_est)
#diff_corn = np.subtract(trans_corn,pos_origin_cam)
mse = np.mean(np.abs(valid_diff[:,range(start,train_dim[0])]), axis = 1)
#mse_corn = np.mean(np.square(diff_corn[:,range(start,train_dim[0])]), axis = 1)
#hor_sum = np.sum(diff[:,range(start,train_dim[0])], axis = 1)
print(mse)
#print(mse_corn)

#storing world coords in hdf5
current_dir = '/home/gaetan/data/hdf5/correct_baselink_gt/'
print(len(hdf5_data["gt_cam_pos"]))
print(len(hdf5_data["est_cam_pos"]))

def dump(output_dir,hdf5_data,ep):
        print('stored data in',output_dir)
        output_hdf5_path = output_dir + '/3D_pos_mask' + '.hdf5'
        hdf5_file = h5py.File(output_hdf5_path, "a")
        episode_group = hdf5_file.create_group(str(ep))
        for sensor_name in hdf5_data.keys():
            episode_group.create_dataset(
                sensor_name, data=np.stack(hdf5_data[sensor_name])
            )
        hdf5_file.close()
dump(current_dir,hdf5_data,'3D_pos')




