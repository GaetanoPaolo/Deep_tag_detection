from matplotlib import pyplot as plt
import cv2
import os
import numpy as np
import h5py

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
observed_pos = 200
for i in range(0,it_len):
    #transforming the logo corner coordinates from world to camera axes
    #ASSUMPTION: The roation matrix derived from quaternions transforms to a z
    #axis that points upward from the drone surface => to be verified
    q = quat[i,:]
    p = pos[i,:]
    qw = q[0]
    qx = q[1]
    qy = q[2]
    qz = q[3]
    # qx = q[0]
    # qy = q[1]
    # qz = q[2]
    # qw = q[3]
    a11 = -2*(qy**2+qz**2)+1
    a12 = 2*qx*qy-2*qz*qw
    a13 = 2*qx*qz+2*qy*qw
    a21 = 2*qx*qy+2*qz*qw
    a22 = -2*(qx**2+qz**2)+1
    a23 = 2*(qy*qz-qx*qw)
    #if tha assumption holds, the 3rd row of the rotationmatrix has to be sign inverted
    a31 = 2*(qx*qz-qy*qw)*(-1)
    a32 = 2*(qy*qz+qw*qx)*(-1)
    a33 = (-2*(qx**2+qy**2)+1)*(-1)
    T_hom = np.matrix([[a11, a12, a13, p[0]],
                      [a21, a22, a23, p[1]],
                      [a31, a32, a33, p[2]],
                      [0,0,0,1]])
    c_cam = np.matmul(T_hom,c_world)
    for j in range(0,c_size[1]):
        #c_cam[:,j] = c_cam[:,j]/c_cam[3,j]
        c_cam[0:2,j]=c_cam[0:2,j]/c_cam[2,j]
    if i == observed_pos:
        print(q)
        print(p)
        norm = np.sqrt(q[0]**2+q[1]**2+q[2]**2+q[3]**2)
        print(norm)
        print(T_hom)
        print(c_cam)
    #project the points in camera space to the image plane
    K = np.array([[100,0,img_size[2]/2],
                 [0,100,img_size[1]/2],
                 [0,0,1]])
    c_proj = np.matmul(K,c_cam[0:3,:])
    proj_array.append(c_proj[0:2,:])
print(proj_array[observed_pos])

