import numpy as np
import matplotlib.pyplot as plt
import h5py

f = h5py.File('/home/gaetan/data/hdf5/correct_baselink_gt/3D_pos_mask.hdf5', 'r+')
base_items = list(f.items())
dset = f.get('3D_pos')
est_cam_pos = np.array(dset.get('est_cam_pos'))
gt_cam_pos = np.array(dset.get('gt_cam_pos'))
print(est_cam_pos.shape)
print(gt_cam_pos.shape)
size = est_cam_pos.shape
#ax = plt.axes(projection = "3d")
for i in range(1,size[0]):
    plt.scatter(est_cam_pos[i,0],est_cam_pos[i,1],s=4.0,c='r')
    plt.scatter(gt_cam_pos[i,0],gt_cam_pos[i,1],s=4.0,c='g')
    print(i)
    print('estimate')
    print(est_cam_pos[i,:])
    print('gt')
    print(gt_cam_pos[i,:])
    plt.pause(2)
plt.show()
