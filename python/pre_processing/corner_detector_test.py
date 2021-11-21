from matplotlib import pyplot as plt
import cv2 as cv
import os
import numpy as np
import h5py
import transform_mat
import time

# load the necessary data 
f = h5py.File('/home/gaetan/data/hdf5/rec_all_topics/data4_preproc2.hdf5', 'r+')
base_items = list(f.items())
#print('Groups:',base_items)
dset = f.get('preproc_data')
#print('Items in group preproc_data',list(dset.items()))
imgs = np.array(dset.get('observation'))
observed_pos = 560
src = imgs[observed_pos,:,:,:]*255


def cornerHarris(val,src):
    thresh = val
    src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
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
    corners_det = []
    corners_valid = []
    print(dst_norm.shape)
    # Drawing a circle around corners
    for i in range(dst_norm.shape[0]):
        for j in range(dst_norm.shape[1]):
            if int(dst_norm[i,j]) > thresh:
                corners_det.append([i,j])
                cv.circle(dst_norm_scaled, (j,i), 5, (0), 2)
    # Showing the result
    print(corners_det)
    cv.imshow('Detected corners',dst_norm_scaled)
    if len(corners_det) > 4:            
        for k in range(0,len(corners_det)):
            cur_pos = corners_det[k]
            for l in range(cur_pos[1]-10,cur_pos[1]+10):
                if np.round(sum(src[cur_pos[0],l,:])) <= np.round(sum([0.00392,0.545,0.00392]*255)):
                        corners_valid.append(cur_pos)
                        break
            #time.sleep(2)
        print(corners_valid)
        corners_unique = []
        for m in range(0,len(corners_valid)):
            cur_pos = corners_valid[m]
            if m > 0:
                for n in range(0,m):
                    prev_pos = corners_valid[n]
                    if abs(cur_pos[0]-prev_pos[0]) > 30 or abs(cur_pos[1]-prev_pos[1]) > 30:
                        if m-n == 1:
                            corners_unique.append(corners_valid[m])
                    else:
                        break
            else:
                corners_unique.append(corners_valid[m])
                    
    return corners_unique


corn = cornerHarris(140,src)
cv.waitKey()
print(corn)
cur_img = imgs[observed_pos,:,:,:]
red = [1,0,0]
for i in range(0,len(corn)):
    cur_img[corn[i][0],corn[i][1],:] = red
myplot = plt.imshow(cur_img)
plt.title('Image'+str(observed_pos))
plt.show()
