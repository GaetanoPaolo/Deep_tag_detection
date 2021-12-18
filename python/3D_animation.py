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
print(est_orig_pos.shape)
print(gt_orig_pos.shape)
size = est_cam_pos.shape
#ax = plt.axes(projection = "3d")
for i in range(1,size[0]):
    plt.scatter(est_orig_pos[i,0],est_orig_pos[i,1],s=4.0,c='r')
    plt.scatter(gt_orig_pos[i,0],gt_orig_pos[i,1],s=4.0,c='g')
    print(i)
    print('estimate')
    print(est_cam_pos[i,:])
    print('gt')
    print(gt_cam_pos[i,:])
    plt.pause(0.0001)
plt.show()
