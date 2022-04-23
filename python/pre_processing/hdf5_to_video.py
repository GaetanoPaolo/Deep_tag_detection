import cv2 as cv
import h5py
import numpy as np

f = h5py.File('/media/gaetan/One Touch/hdf5/psi_800res_alt_rot_concrete_fine/data4_sync.hdf5', 'r+')
#f = h5py.File('/home/gaetan/data/hdf5/psi_200res_alt/data4_sync.hdf5', 'r+')
base_items = list(f.items())
print(base_items)
dset = f.get('1')
imgs = np.array(dset.get('observation'))
print('image amount')
print(imgs.shape)
height, width, layers = imgs.shape[1:4]
print(height)
fourcc = cv.VideoWriter_fourcc(*'MP4V')
video = cv.VideoWriter('psi_800res_alt_rot_concrete_fine.mp4', fourcc , 30.0, (width,height),True)

for seq in range(1,len(base_items)):
    dset = f.get(str(seq))
    imgs = np.array(dset.get('observation'))
    frame_len = imgs.shape[0]
    for frame in range(0,frame_len):
        video.write(np.uint8(imgs[frame,:,:,:]*255))

cv.destroyAllWindows()
video.release()