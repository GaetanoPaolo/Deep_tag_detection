from matplotlib import pyplot as plt
import cv2 as cv
import os
import numpy as np
import h5py
import transform_mat

# load the necessary data 
f = h5py.File('/home/gaetan/data/hdf5/rec_all_topics/data4_preproc2.hdf5', 'r+')
base_items = list(f.items())
#print('Groups:',base_items)
dset = f.get('preproc_data')
#print('Items in group preproc_data',list(dset.items()))
imgs = np.array(dset.get('observation'))
observed_pos = 1000
src = imgs[observed_pos,:,:,:]*255
src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

def cornerHarris_demo(val):
    thresh = val
    # Detector parameters
    blockSize = 2
    apertureSize = 3
    k = 0.04
    # Detecting corners
    dst = cv.cornerHarris(src_gray, blockSize, apertureSize, k)
    # Normalizing
    dst_norm = np.empty(dst.shape, dtype=np.float32)
    cv.normalize(dst, dst_norm, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
    dst_norm_scaled = cv.convertScaleAbs(dst_norm)
    print(dst_norm.shape)
    # Drawing a circle around corners
    for i in range(dst_norm.shape[0]):
        for j in range(dst_norm.shape[1]):
            if int(dst_norm[i,j]) > thresh:
                cv.circle(dst_norm_scaled, (j,i), 5, (0), 2)
    # Showing the result
    cv.imshow('Detected corners',dst_norm_scaled)
cornerHarris_demo(140)
cv.waitKey()