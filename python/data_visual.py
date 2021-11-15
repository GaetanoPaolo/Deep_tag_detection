from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import cv2
import os
import time
import numpy as np
import h5py
f = h5py.File('/home/gaetan/data/hdf5/test_sim_flight/data4_preproc2.hdf5', 'r')
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
current_pos = 200
myplot = plt.imshow(imgs[current_pos,:,:,:])
print(imgs[current_pos,10,10,:])
plt.title('Current position'+str(pos[current_pos,:]))
plt.show()

cur_quat = quat[current_pos,:]
norm = np.sqrt(cur_quat[0]**2+cur_quat[1]**2+cur_quat[2]**2+cur_quat[3]**2)
print(norm)
#print(np.column_stack((image_time[200:400,:],pose_time[200:400,:],pos[200:400,:])))
# for i in range(300,600):
#     print(str(image_time[i,:])+str(pose_time[i,:])+str(pos[i,:]))
#print(pose_time[300:350,1])    
