from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import cv2
import os
import time
import numpy as np
import h5py
import det_logo_image
f = h5py.File('/home/gaetan/data/hdf5/rec_all_topics/data4_labelled.hdf5', 'r')
base_items = list(f.items())
print('Groups:',base_items)
dset = f.get('labelled_data')
print('Items in group preproc_data',list(dset.items()))
imgs = np.array(dset.get('observation'))
pos = np.array(dset.get('position'))
quat = np.array(dset.get('orientation'))
#rel_pos = np.array(dset.get('relative_position'))
image_time = np.array(dset.get('image_time'))
pose_time = np.array(dset.get('pose_time'))
corn = np.array(dset.get('corners'))
size = imgs.shape

#displaying indicated image + corresponding position
current_pos = 1000
cur_img = imgs[current_pos,:,:,:]
cur_corn = corn[current_pos,:,:]
red = [1,0,0]
size_corn = cur_corn.shape
for i in range(0,size_corn[0]):
    cur_img[cur_corn[i,0],cur_corn[i,1],:] = red
myplot = plt.imshow(cur_img)
print(det_logo_image.logo_image(True,imgs[current_pos,:,:,:]))
plt.title('Current position'+str(pos[current_pos,:]))
plt.show()

   
