from matplotlib import pyplot as plt
import cv2
import os
import numpy as np
import h5py
import transform_mat

# load the necessary data 
f = h5py.File('/home/gaetan/data/hdf5/test_sim_flight/data4_preproc2.hdf5', 'r')
base_items = list(f.items())
print('Groups:',base_items)
dset = f.get('preproc_data')
print('Items in group preproc_data',list(dset.items()))
imgs = np.array(dset.get('observation'))
rel_pos = np.array(dset.get('relative_position'))
pos = np.array(dset.get('position'))
quat = np.array(dset.get('orientation'))
image_time = np.array(dset.get('image_time'))
pose_time = np.array(dset.get('pose_time'))
pos_size = pos.shape
it_len = image_time.shape[0]

# define the world corner logo positions 
#for the current dataset
abs_y_max = 0.25/2
abs_x_max = 0.350479/2
z = 0.001
c_world = np.array([[abs_x_max,-abs_x_max,-abs_x_max,abs_x_max],
                     [-abs_y_max,-abs_y_max,abs_y_max,abs_y_max],
                     [z,z,z,z],
                     [1,1,1,1]])
c_size = c_world.shape
img_size = imgs.shape
proj_array = []

#Define the observed image array index
#List of indices with visible logo: 200,430,450,470,1300,1310
observed_pos = 200

#calculating focal lengths of camera
horizontal_field_of_view = (80 * img_size[2]/img_size[1]) * 3.14 / 180
vertical_field_of_view = 80 * 3.14 / 180
fx = -img_size[2]/2*np.tan(horizontal_field_of_view/2)**(-1)
fy = -img_size[1]/2*np.tan(vertical_field_of_view/2)**(-1)
print([fx,fy])

#calculating static transformation matrices on drone itself
T_baselink_downlink = transform_mat.transf_mat([0,0,0,1],[0.05,0,-0.1])
for i in range(0,it_len):
    q = quat[i,:]
    p = pos[i,:]
    T_world_baselink = transform_mat.transf_mat(q,p)
    c_baselink = np.matmul(T_world_baselink,c_world)
    c_cam = np.matmul(T_baselink_downlink,c_baselink)
    for j in range(0,c_size[1]):
        c_cam[0:3,j]=c_cam[0:3,j]/c_cam[2,j]
    if i == observed_pos:
        print(q)
        print(p)
        norm = np.sqrt(q[0]**2+q[1]**2+q[2]**2+q[3]**2)
        print(norm)
        print(T_world_baselink)
        print(T_baselink_downlink)
        print(c_cam)
    #project the points in camera space to the image plane
    K = np.array([[fx,0,img_size[2]/2],
                 [0,fy,img_size[1]/2],
                 [0,0,1]])
    c_proj = np.matmul(K,c_cam[0:3,:])
    proj_array.append(c_proj[0:2,:])
print(proj_array[observed_pos])

#setting the corner points of the logo to red color
cur_img = imgs[observed_pos,:,:,:]
for j in range(0,c_size[1]):
    red = [1,0,0]
    cur_proj = proj_array[observed_pos]
    #setting the current image pixel to red
    if cur_proj[0,j] < img_size[1] and cur_proj[1,j] < img_size[0]: 
        x_pix = int(np.round(cur_proj[0,j],0))
        y_pix = int(np.round(cur_proj[1,j],0))
        cur_img[x_pix,y_pix,:] = red
    print(x_pix,y_pix)
#display the modified image
myplot = plt.imshow(cur_img)
plt.title('Current position'+str(pos[observed_pos,:]))
plt.show()