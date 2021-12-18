import numpy as np
import matplotlib.pyplot as plt
import h5py

f = h5py.File('/home/gaetan/data/hdf5/correct_baselink_gt/3D_pos_mask.hdf5', 'r+')
base_items = list(f.items())
dset = f.get('3D_pos')
est_cam_pos = np.array(dset.get('est_cam_pos'))
gt_cam_pos = np.array(dset.get('gt_cam_pos'))
gt_orig_pos = np.array(dset.get('gt_orig_pos'))
est_orig_pos = np.array(dset.get('est_orig_pos'))
imgs = np.array(dset.get('observation'))
print(imgs.shape)
print(est_orig_pos.shape)
print(gt_orig_pos.shape)
size = est_cam_pos.shape
img_size = imgs.shape
horizontal_field_of_view = (80 * img_size[2]/img_size[1]) * 3.14 / 180
vertical_field_of_view = 80 * 3.14 / 180
fx = -img_size[2]/2*np.tan(horizontal_field_of_view/2)**(-1)
fy = -img_size[1]/2*np.tan(vertical_field_of_view/2)**(-1)
K = np.array([[-fx,0,img_size[2]/2],
                 [0,-fy,img_size[1]/2],
                 [0,0,1]])
count = 0
#ax = plt.axes(projection = "3d")
for i in range(1,size[0]):
    #est_norm = est_orig_pos[i,0:3]/est_orig_pos[i,2]
    est_norm = est_orig_pos[i,0:3]/est_orig_pos[i,2]
    gt_norm = gt_orig_pos[i,0:3]/gt_orig_pos[i,2]
    est_proj = np.matmul(K, np.transpose(est_norm))
    gt_proj = np.matmul(K, np.transpose(gt_norm))
    plt.xlim(0,200)
    plt.ylim(0,200)
    plt.imshow(imgs[i,:,:,:], cmap='gray',vmin=0, vmax=255)
    plt.scatter(est_proj[0],est_proj[1],s=4.0,c='r')
    plt.scatter(gt_proj[0],gt_proj[1],s=4.0,c='b')
    plt.pause(0.000001)
    count += 1
    if count == 20:
        plt.clf()
        count = 0
plt.show()