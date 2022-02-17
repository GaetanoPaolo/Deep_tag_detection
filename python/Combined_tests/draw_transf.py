import cv2 as cv
import os
import numpy as np

#creating bounding box around image keypoints
def findBB_rot(keypts):
    rect = cv.minAreaRect(np.array(keypts))
    box = cv.boxPoints(rect)
    box = np.int0(box)
    return box
def findBB_straight(keypts):
    x,y,w,h = cv.boundingRect(np.array(keypts))
    box = np.array([(x,y),(x+w,y),(x+w,y+h),(x,y+h)])
    box = np.int0(box)
    return box
def rearrange_target_BB(tempBB,targetBB,target_keypts,temp_keypts):
    corr_target_pts = []
    size_tempBB = tempBB.shape
    BB_target = np.zeros(size_tempBB)
    size_targetBB = targetBB.shape
    size_targetpts = len(target_keypts)
    size_temppts = len(temp_keypts)
    for k in range(0,size_tempBB[0]):
        diff_temp = []
        for l in range(0,size_temppts):
            diff_temp.append(((tempBB[k,:]-np.array(temp_keypts[l]))**2).mean(axis=0))
        closest_pos = diff_temp.index(min(diff_temp))
        corr_target_pts.append(target_keypts[closest_pos])
    # corr_terget_pts list of keypoints closest to each corner of the template bounding bow in order
    diff_mat = np.zeros((size_targetBB[0],len(corr_target_pts)))
    for w in range(0,size_targetBB[0]):
        diff_target = []
        for m in range(0,len(corr_target_pts)):
            diff_target.append(((targetBB[w,:]-np.array(corr_target_pts[m]))**2).mean(axis=0))
        diff_mat[w,:] = np.array(diff_target)
    tot_max = np.amax(diff_mat)
    for n in range(0,size_targetBB[0]):
        cur_min = np.amin(diff_mat)
        pos = np.where(diff_mat == cur_min)
        #the column defines the correct position, the row defines the corresponding target BB corner
        BB_target[pos[1],:] = targetBB[pos[0],:]
        diff_mat[pos[0],:] = tot_max
        diff_mat[:,pos[1]] = tot_max
    return np.array(BB_target)
def world_coord(pts,size,rot):
    #size = pixel dimensions of the template
    dim = pts.shape
    vert_ratio = 0.985/max(size)
    horiz_ratio = 0.68/min(size)
    #inverting order of position indices due to the x direction being along the long side
    #and y being along the short side in gazebo
    x_origin = round(max(size)/2)
    #y_origin = round(max(size)/2)
    y_origin = round(min(size)/2)
    if rot == 0:
        pt_origin = np.array([int(x_origin), int(y_origin)])
        world_pts = []
        for i in range(0,dim[0]):
            rel_pos_px = pt_origin - np.array([pts[i,0],pts[i,1]])
            rel_pos_3d = (rel_pos_px[0]*vert_ratio,rel_pos_px[1]*horiz_ratio,0.001)
            world_pts.append(rel_pos_3d)
    elif rot == 90:
        pt_origin = np.array([int(y_origin), int(-x_origin)])
        world_pts = []
        for i in range(0,dim[0]):
            rel_pos_px = pt_origin - np.array([pts[i,0],-pts[i,1]])
            rel_pos_3d = (rel_pos_px[0]*vert_ratio,rel_pos_px[1]*horiz_ratio,0.001)
            world_pts.append(rel_pos_3d)
    elif rot == 180:
        pt_origin = np.array([int(-x_origin), int(-y_origin)])
        world_pts = []
        for i in range(0,dim[0]):
            rel_pos_px = pt_origin - np.array([-pts[i,0],-pts[i,1]])
            rel_pos_3d = (rel_pos_px[0]*vert_ratio,rel_pos_px[1]*horiz_ratio,0.001)
            world_pts.append(rel_pos_3d)
    elif rot == 270:
        pt_origin = np.array([int(-y_origin), int(x_origin)])
        world_pts = []
        for i in range(0,dim[0]):
            rel_pos_px = pt_origin - np.array([-pts[i,0],pts[i,1]])
            rel_pos_3d = (rel_pos_px[0]*vert_ratio,rel_pos_px[1]*horiz_ratio,0.001)
            world_pts.append(rel_pos_3d)
    return np.array(world_pts)