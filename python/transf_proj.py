from matplotlib import pyplot as plt
import cv2
import os
import numpy as np
import h5py

# load the necessary data 
f = h5py.File('/home/gaetan/data/hdf5/test_sim_flight/data4_preproc.hdf5', 'r')
base_items = list(f.items())
print('Groups:',base_items)
dset = f.get('0')
print('Items in group 0',list(dset.items()))
imgs = np.array(dset.get('observation'))
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
c11_world = np.array([abs_x_max,-abs_y_max,z])
c12_world = np.array([-abs_x_max,-abs_y_max,z])
c21_world = np.array([-abs_x_max,abs_y_max,z])
c22_world = np.array([abs_x_max,abs_y_max,z])
for i in range(0,it_len):
    q = quat[i,:]
    p = pos[i,:]
    q0 = q[0]
    q1 = q[1]
    q2 = q[2]
    q3 = q[3]
    a11 = 2*(q0**2+q1**2)-1
    a12 = 2*q1*q2+2*q0*q3
    a13 = 2*q1*q3-2*q0*q2
    a21 = 2*q1*q2-2*q0*q3
    a22 = 2*(q0**2+q2**2)-1
    a23 = 2*(q2*q3+q0*q1)
    a31 = 2*(q1*q3+q0*q2)
    a32 = 2*(q2*q3-q0*q1)
    a33 = 2*(q0**2+q3**2)-1
    T_hom = np.matrix([a11, a12, a13, 0],
                      [a21, a22, a23, 0]
                      [a31, a32, a33, 0]
                      [p[0],p[1],p[3],1])
    


