from matplotlib import pyplot as plt
import cv2 as cv
import numpy as np
import h5py
import crop
import itertools
import draw_transf as dw

#load the logo template
logo_temp = cv.imread('/home/gaetan/code/simulation_ws/src/my_simulations/models/psi_logo/materials/textures/poster-psi-drone-logo-2percent.png',0)
rot = 0
#currently only using vertical version as rotations don't seem to improve the matching
#removing the blue edge of the logo template to detect logo itself
logo_temp = crop.crop_img(logo_temp)
# load the labelled gazebo data
f = h5py.File('/home/gaetan/data/hdf5/logo_correct_size/data4_labelled.hdf5', 'r+')
base_items = list(f.items())
dset = f.get('labelled_data')
imgs = np.array(dset.get('observation'))
corn = np.array(dset.get('corners'))
pos = np.array(dset.get('position'))
#initiating necessary parameters
fx = 100
fy = 100
arr_size = pos.shape
img_size = imgs[0,:,:,:].shape
K = np.array([[fx,0,img_size[1]/2],
                [0,fy,img_size[0]/2],
                [0,0,1]])
#code source till line 31: https://datahacker.rs/feature-matching-methods-comparison-in-opencv/
#creating keypoint matchers and finders
orb = cv.ORB_create()
bf = cv.BFMatcher_create(cv.NORM_HAMMING,crossCheck=True)
#calculating template keypoints and descriptors before the loop
kp_temp, des_temp = orb.detectAndCompute(logo_temp,None)
train_dim = imgs.shape
trans_est = np.zeros((3,arr_size[0]))
start = 0
count = 0
for k in range(100,train_dim[0]):
    src = imgs[k,:,:,:]*255
    src_gray = np.uint8(cv.cvtColor(src, cv.COLOR_BGR2GRAY))
    kp_target, des_target = orb.detectAndCompute(src_gray,None)
    if type(des_target) != type(None):
        if start == 0:
            start = k
        matches = bf.match(des_temp, des_target)
        matches = sorted(matches,key=lambda x:x.distance)
        pos_temp = []
        pos_target = []
        for m in range(0,len(matches)):
            target_pt_idx = matches[m].trainIdx
            temp_pt_idx = matches[m].queryIdx
            temp_coord = kp_temp[temp_pt_idx].pt
            target_coord = kp_target[target_pt_idx].pt
            pos_temp.append(temp_coord)
            pos_target.append(target_coord)
        pos_temp_hom = np.float32(pos_temp).reshape(-1,1,2)
        pos_target_hom = np.float32(pos_target).reshape(-1,1,2)
        M, mask = cv.findHomography( pos_temp_hom, pos_target_hom,cv.RANSAC,0.1)
        matchesMask = mask.ravel().tolist()
        inlier_matches = []
        rem = 0
        for k in range(0,len(matches)):
            if matchesMask[k] == 1:
                inlier_matches.append(matches[k])
            else:
                pos_temp.pop(k-rem)
                pos_target.pop(k-rem)
                rem += 1
        if len(inlier_matches) > 3:
            pos_temp_world = dw.world_coord(np.array(pos_temp),logo_temp,rot)
            dist_coeffs = np.zeros((4,1))
            (suc,est_rot,est_trans) = cv.solvePnP(pos_temp_world, np.array(pos_target), K, dist_coeffs, flags=cv.SOLVEPNP_ITERATIVE)
            trans_est[:,k] = est_trans[:,0]
        else:
            trans_est[:,k] = trans_est[:,k-1]

pos = np.transpose(pos)
diff = np.subtract(pos,trans_est)
mse = np.mean(np.square(diff[:,range(start,train_dim[0])]), axis = 1)
hor_sum = np.sum(diff[:,range(start,train_dim[0])], axis = 1)
print(mse)


