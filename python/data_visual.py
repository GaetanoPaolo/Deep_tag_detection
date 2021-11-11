from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import cv2
import os
import time
import numpy as np
import h5py
f = h5py.File('/home/gaetan/data/hdf5/test_sim_flight/data4_preproc.hdf5', 'r')
base_items = list(f.items())
print('Groups:',base_items)
#dset = f.get('preproc_data')
dset = f.get('preproc_data')
print('Items in group preproc_data',list(dset.items()))
imgs = np.array(dset.get('observation'))
pos = np.array(dset.get('position'))
quat = np.array(dset.get('orientation'))
#rel_pos = np.array(dset.get('relative_position'))
image_time = np.array(dset.get('image_time'))
pose_time = np.array(dset.get('pose_time'))
size = imgs.shape

#displaying indicated image + corresponding position
current_pos = 0
myplot = plt.imshow(imgs[current_pos,:,:,:])
plt.title('Current position'+str(quat[current_pos,:]))
plt.show()

cur_quat = quat[current_pos,:]
norm = np.sqrt(cur_quat[0]**2+cur_quat[1]**2+cur_quat[2]**2+cur_quat[3]**2)
print(norm)

#print(rel_pos[0:50,:])
    
