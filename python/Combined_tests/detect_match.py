
import cv2 as cv
import numpy as np
import time
import crop
import itertools
import draw_transf as dw
import transform_mat as tm

def detect_match(K,kp_temp,des_temp,temp_shape,detector,bf,src_gray):
    #keep only a square inside the round camera capture to avoid detections at the edge of the lens
    src_gray = src_gray[144:657,144:705]

    kp_target, des_target = detector.detectAndCompute(src_gray,None)
    matches = bf.match(des_temp, des_target)
    matches = sorted(matches,key=lambda x:x.distance)

    pos_temp = []
    pos_target = []
    matched_temp_kpts = []
    matched_target_kpts = []
    for m in range(0,len(matches)):
        target_pt_idx = matches[m].trainIdx
        temp_pt_idx = matches[m].queryIdx
        temp_coord = kp_temp[temp_pt_idx].pt
        matched_temp_kpts.append(kp_temp[temp_pt_idx])
        target_coord = kp_target[target_pt_idx].pt
        target_coord = (target_coord[0]+144,target_coord[1]+144)
        matched_target_kpts.append(kp_target[target_pt_idx])
        pos_temp.append(temp_coord)
        pos_target.append(target_coord)
    #apply solvepnp to estimate translation: https://learnopencv.com/head-pose-estimation-using-opencv-and-dlib/
    pos_temp_world = dw.world_coord(np.array(pos_temp),temp_shape,0)
    dist_coeffs = np.zeros((4,1))
    (suc,rot,trans,inliers) = cv.solvePnPRansac(pos_temp_world, np.array(pos_target), K, dist_coeffs, flags=cv.SOLVEPNP_ITERATIVE, iterationsCount=2000, reprojectionError=8)
    if type(inliers) == 'NoneType':
        trans = np.zeros((3,1))
    return trans