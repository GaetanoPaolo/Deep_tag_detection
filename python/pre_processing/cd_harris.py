import cv2 as cv
import os
import numpy as np
# This function 
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
    corners_unique = []
    # Getting corner coordinates
    for i in range(dst_norm.shape[0]):
        for j in range(dst_norm.shape[1]):
            if int(dst_norm[i,j]) > thresh:
                corners_det.append([i,j])
    # Filtering out the corners inside the logo and the adjacent detections,
    #in order to keep one point per corner of the rectangle
    if len(corners_det) > 4:            
        for k in range(0,len(corners_det)):
            cur_pos = corners_det[k]
            if 9 < cur_pos[1] < 191:
                for l in range(cur_pos[1]-10,cur_pos[1]+10):
                    if np.round(sum(src[cur_pos[0],l,:])) <= np.round(sum([0.00392,0.545,0.00392]*255)):
                            corners_valid.append(cur_pos)
                            break
            elif 9 < cur_pos[0] < 191:
                for l in range(cur_pos[0]-10,cur_pos[0]+10):
                    if np.round(sum(src[l,cur_pos[1],:])) <= np.round(sum([0.00392,0.545,0.00392]*255)):
                                corners_valid.append(cur_pos)
                                break
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