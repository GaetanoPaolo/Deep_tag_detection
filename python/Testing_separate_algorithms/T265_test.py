from matplotlib import pyplot as plt
import cv2 as cv
import os
import numpy as np
import h5py
import time
import crop
import itertools
import draw_transf as dw
import transform_mat as tm
#load the logo template
logo_temp = cv.imread('/home/gaetan/code/simulation_ws/src/my_simulations/models/psi_logo/materials/textures/poster-psi-drone-logo-2percent.png',0)
logo_temp_color = cv.imread('/home/gaetan/code/simulation_ws/src/my_simulations/models/psi_logo/materials/textures/poster-psi-drone-logo.png')
#The scale var indicates the percentage kept from the original logo resolution
scale = 1
rot = 0
plt.imshow(logo_temp),plt.show()
logo_temp = crop.crop_img(logo_temp,scale)
# load the camera parameters stored in episode 1
f = h5py.File('/home/gaetan/data/hdf5/T265/data4.hdf5', 'r+')
base_items = list(f.items())
print(base_items)
dset1 = f.get('0')
group_items = list(dset1.items())
print(group_items)
#imgs = np.array(dset1.get('observation'))
pos = np.array(dset1.get('pos_est'))
quat = np.array(dset1.get('orientation'))
rel_pos = np.array(dset1.get('relative_position'))
img_stamp = np.array(dset1.get('image_time'))
pose_stamp = np.array(dset1.get('est_time'))
observed_pos = 800
#src = imgs[observed_pos,:,:,:]*255
#src_gray = np.uint8(src)
#plt.imshow(imgs[observed_pos,:,:,:]),plt.show()
print(img_stamp.shape)
print(pose_stamp.shape)
merg = np.zeros((1461,4))
merg[:,0:2] = img_stamp[0:1461,:]
merg[:,2:4] = pose_stamp
print(merg[0:100,:])
