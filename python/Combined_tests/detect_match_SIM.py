import cv2 as cv
import numpy as np
import time
import crop
import itertools
import draw_transf as dw
import transform_mat as tm

def detect_match(K,kp_temp,des_temp,temp_shape,detector,bf,src_gray):
    #keep only a square inside the round camera capture to avoid detections at the edge of the lens
    zer_res = np.array([[0.0],
                    [0.0],
                    [0.0]])
    trans = zer_res
    kp_target, des_target = detector.detectAndCompute(src_gray,None)
    try:
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
            matched_target_kpts.append(kp_target[target_pt_idx])
            pos_temp.append(temp_coord)
            pos_target.append(target_coord)
        #apply solvepnp to estimate translation: https://learnopencv.com/head-pose-estimation-using-opencv-and-dlib/
        pos_temp_world = dw.world_coord(np.array(pos_temp),temp_shape,0)
        dist_coeffs = np.zeros((4,1))
        if len(pos_temp) > 3 and len(pos_target) > 3:
            (suc,rot,trans,inliers) = cv.solvePnPRansac(pos_temp_world, np.array(pos_target), K, dist_coeffs, flags=cv.SOLVEPNP_ITERATIVE, iterationsCount=2000, reprojectionError=2.0)
            if isinstance(inliers,type(None)):
                trans = zer_res
            elif len(inliers) < 20:
                trans = zer_res
        return trans
    except:
        return trans
def resolution_sel(est_orb_lst,rel_pos_c):
    alt_diff = []
    for i in range(0,len(est_orb_lst)):
        cur_est = est_orb_lst[i]
        cur_diff = abs(rel_pos_c[2]-cur_est[2,0])
        alt_diff.append(cur_diff)
    min_alt_diff = min(alt_diff)
    min_index = alt_diff. index(min_alt_diff) 
    
    #extending the orb estimate with the best resolution
    res_list = [1,2,5,10,100]
    ext_orb_res = np.zeros((4,1))
    ext_orb_res[0:3,0] = np.squeeze(est_orb_lst[min_index],axis = 1)
    ext_orb_res[3,0] = res_list[min_index]
    return ext_orb_res