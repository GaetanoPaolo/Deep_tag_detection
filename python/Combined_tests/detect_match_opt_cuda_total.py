
import cv2 as cv
import numpy as np
import time
import crop
import itertools
import draw_transf as dw
import transform_mat as tm
import solvepnp_gpu
from matplotlib import pyplot as plt
def detect_match(K,kp_temp,des_temp,temp_shape,kp_target,des_target,bf,src_gray,logo_temp):
    #keep only a square inside the round camera capture to avoid detections at the edge of the lens
    src_gray = src_gray[144:657,144:705]
    zerarr = np.array([[0.0],
                            [0.0],
                            [0.0]])
    #kp_target, des_target = detector.detectAndCompute(src_gray,None)
    #try:
    # des_temp_cuda = cv.cuda_GpuMat()
    # des_target_cuda = cv.cuda_GpuMat()
    # des_temp_cuda.upload(des_temp)
    # des_target_cuda.upload(des_target)
    print('Kp amounts')
    print(len(kp_temp))
    print(len(kp_target))
    matches_cuda = bf.matchAsync(des_temp, des_target)
    #matches = matches_cuda.download()
    matches = bf.matchConvert(matches_cuda)
    matches = sorted(matches,key=lambda x:x.distance)
    ORB_matches =cv.drawMatches(logo_temp, kp_temp, src_gray, kp_target, matches[0:len(matches)], None, flags=2)
    plt.imshow(ORB_matches),plt.show()
    print('Match amount')
    print(len(matches))
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
    if len(pos_temp) < 4 or len(pos_target) < 4:
        trans = zerarr
        inliers = 0.0
    else:
        pos_temp_world, world_pts = dw.world_coord(np.array(pos_temp),temp_shape,0)
        dist_coeffs = np.zeros((4,1))
        solve_res = solvepnp_gpu.solvepnp_gpu(200,0.02,30,len(pos_target),pos_temp_world, np.array(pos_target,dtype=np.float32), K, dist_coeffs)
        trans = np.reshape(np.array(solve_res[0:3]),(3,1))
        inliers = solve_res[3]
        if inliers == 0.0:
            trans = zerarr
        # elif inliers < 20.0:
        #     trans = zerarr
# except:
#     return zerarr,[]
    print(inliers)
    print(trans)
    return trans,inliers

def resolution_sel(est_orb_lst,rel_pos_c):
    alt_diff = []
    for i in range(0,len(est_orb_lst)):
        cur_est = est_orb_lst[i]
        cur_diff = abs(rel_pos_c[2,0]-cur_est[2,0])
        alt_diff.append(cur_diff)
    min_alt_diff = min(alt_diff)
    min_index = alt_diff. index(min_alt_diff) 
    
    #extending the orb estimate with the best resolution
    res_list = [2,5,10]
    ext_orb_res = np.zeros((4,1))
    ext_orb_res[0:3,0] = np.squeeze(est_orb_lst[min_index],axis = 1)
    ext_orb_res[3,0] = res_list[min_index]
    return ext_orb_res
def mid_pos(pt1,pt2):
    midy = (pt1[0]+pt2[0])/2
    midx = (pt1[1]+pt2[1])/2
    return [midy,midx]

def rectangle_side_fractal(corn_list,des_len):
    out_lst = []
    for i in range(0,len(corn_list)):
        corn_list_2 = list(np.roll(np.array(corn_list),-i,axis = 0))
        temp_out_lst = []
        temp_out_lst.append(corn_list_2[0])
        temp_out_lst.append(mid_pos(corn_list_2[0],corn_list_2[1]))
        out_lst = out_lst + temp_out_lst
    if len(out_lst) == des_len:
        return out_lst
    else:
        return rectangle_side_fractal(out_lst,des_len)
def mean_keypoint_dist_temp(kpts,pt):
    diffs = np.zeros((len(kpts),1))
    for m in range(0,len(kpts)):
        cur_pt = kpts[m].pt
        int_pt = np.array([cur_pt[0],cur_pt[1]])
    diff = np.subtract(int_pt,np.array(pt))
    norm = np.linalg.norm(diff)
    diffs[m,0] = norm
    return np.mean(diffs,axis = 0)
def mean_keypoint_dist_target(kpts,pt):
    diff = np.subtract(kpts,np.array(pt))
    norm = np.linalg.norm(diff, axis=1)
    return np.mean(diff,axis = 0)

def kp_preproc(kpts):
    Z = np.zeros((len(kpts),2))
    for m in range(0,len(kpts)):
        cur_pt = kpts[m].pt
        int_pt = np.array([cur_pt[0],cur_pt[1]])
        Z[m,:] = int_pt
    return np.float32(Z)