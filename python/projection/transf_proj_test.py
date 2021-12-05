from matplotlib import pyplot as plt
import cv2 as cv
import os
import numpy as np
import h5py
import transform_mat

# load the necessary data 
f = h5py.File('/home/gaetan/data/hdf5/correct_baselink_gt/data4_labelled.hdf5', 'r')
base_items = list(f.items())
print('Groups:',base_items)
dset = f.get('labelled_data')
print('Items in group preproc_data',list(dset.items()))
imgs = np.array(dset.get('observation'))
rel_pos = np.array(dset.get('relative_position'))
pos = np.array(dset.get('position'))
quat = np.array(dset.get('orientation'))
image_time = np.array(dset.get('image_time'))
pose_time = np.array(dset.get('pose_time'))
pos_base_footprint = np.array(dset.get("pos_base_footprint"))
quat_base_footprint = np.array(dset.get("quat_base_footprint"))
pos_base_stabilized = np.array(dset.get("pos_base_stabilized"))
quat_base_stabilized = np.array(dset.get("quat_base_stabilized"))
pos_base_link = np.array(dset.get("pos_base_link"))
quat_base_link = np.array(dset.get("quat_base_link"))
pos_down_link = np.array(dset.get("pos_down_link"))
quat_down_link = np.array(dset.get("quat_down_link"))
pos_down_optical_frame = np.array(dset.get("pos_down_optical_frame"))
quat_down_optical_frame = np.array(dset.get("quat_down_optical_frame"))
pos_size = pos.shape
it_len = pos.shape[0]

#calculating static transformation matrices on drone itself
T_baselink_downlink = np.linalg.inv(transform_mat.transf_mat(quat_down_link[0,:],pos_down_link[0,:]))
T_downlink_downoptframe = np.linalg.inv(transform_mat.transf_mat(quat_down_optical_frame[0,:],pos_down_optical_frame[0,:]))
T_baselink_downoptframe = np.matmul(transform_mat.transf_mat(quat_down_link[0,:],pos_down_link[0,:]),transform_mat.transf_mat(quat_down_optical_frame[0,:],pos_down_optical_frame[0,:]))
p_downoptframe_baselink = T_baselink_downoptframe[:,3]
# define the world corner logo positions 
#for the current dataset
abs_y_max = 0.68/2
abs_x_max =0.98/2
z = 0.001
c_world = np.array([[abs_x_max,-abs_x_max,-abs_x_max,abs_x_max],
                     [-abs_y_max,-abs_y_max,abs_y_max,abs_y_max],
                     [z,z,z,z],
                     [1,1,1,1]])
c_size = c_world.shape
img_size = imgs.shape
proj_array = []
c0_world = np.array([[0],
                    [0],
                    [0],
                    [1]])
#Define the observed image array index
#List of indices with visible logo: 150,560,1000
observed_pos =300

#calculating focal lengths of camera
horizontal_field_of_view = (80 * img_size[2]/img_size[1]) * 3.14 / 180
vertical_field_of_view = 80 * 3.14 / 180
fx = -img_size[2]/2*np.tan(horizontal_field_of_view/2)**(-1)
fy = -img_size[1]/2*np.tan(vertical_field_of_view/2)**(-1)
print([fx,fy])


for i in range(0,it_len):
    q = quat[i,:]
    p = pos[i,:]
    T_world_baselink = transform_mat.transf_mat(q,p)
    p_downoptframe_world  = np.matmul(T_world_baselink,p_downoptframe_baselink)
    T_world_baselink_inv = np.linalg.inv(T_world_baselink)
    c_baselink = np.matmul(T_world_baselink_inv,c_world)
    c0_baselink = np.matmul(T_world_baselink_inv,c0_world)
    c_downlink = np.matmul(T_baselink_downlink,c_baselink)
    c0_downlink = np.matmul(T_baselink_downlink, c0_baselink)
    c_cam = np.matmul(T_downlink_downoptframe,c_downlink)
    c0_cam= np.matmul(T_downlink_downoptframe, c0_downlink)
    if i == observed_pos:
        print(p)
        print(p_downoptframe_world)
        print(c0_cam)
     #project the points in camera space to the image plane
    K = np.array([[-fx,0,img_size[2]/2],
                 [0,-fy,img_size[1]/2],
                 [0,0,1]])
    for j in range(0,c_size[1]):
        c_cam[0:3,j]=c_cam[0:3,j]/c_cam[2,j]
    c_proj = np.matmul(K,c_cam[0:3,:])
    proj_array.append(c_proj[0:2,:])
print(proj_array[observed_pos])

#setting the corner points of the logo to red color
cur_img = imgs[observed_pos,:,:,:]
cur_proj = proj_array[observed_pos]
keypts = []
for j in range(0,c_size[1]):
    red = [1,0,0]
    #setting the current image pixel to red
    if abs(np.round(cur_proj[0,j])) < img_size[1] and abs(np.round(cur_proj[1,j])) < img_size[2]: 
        x_pix = int(np.round(cur_proj[0,j],0))
        y_pix = int(np.round(cur_proj[1,j],0))
        keypts.append((int(x_pix),int(y_pix)))
cv.drawContours(cur_img,[np.array(keypts)],0,(255,0,0),1)
print(image_time[observed_pos,:])
print(pose_time[observed_pos,:])
#display the modified image
myplot = plt.imshow(cur_img)
plt.title('Current position'+str(pos[observed_pos,:]))
plt.show()