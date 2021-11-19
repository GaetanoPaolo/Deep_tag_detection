from matplotlib import pyplot as plt
import cv2
import os
import numpy as np
import h5py
import time


# load all the necessary data from the hdf5 file
#f = h5py.File('/home/gaetan/data/hdf5/test_sim_flight/data3.hdf5', 'r')
current_dir = '/home/gaetan/data/hdf5/test_sim_flight/'
f = h5py.File(current_dir+'data4.hdf5', 'r')
base_items = list(f.items())
print('Groups:',base_items)
dset = f.get('0')
print('Items in group 0',list(dset.items()))
imgs = np.array(dset.get('observation'))
pos = np.array(dset.get('position'))
quat = np.array(dset.get('orientation'))
image_time = np.array(dset.get('image_time'))
pose_time = np.array(dset.get('pose_time'))
#rel_pos = pos
pos_size = pos.shape

# define new dict for hdf5 storage
hdf5_data = {"observation": [], "position": [], "orientation": [],"relative_position": [],'image_time': [], 'pose_time': []}

#check which timestamps are equal and synchronize images and poses
if image_time.shape[0] < pose_time.shape[0]:
    it_len1 = image_time.shape[0]
    long_time = pose_time
    short_time = image_time
    it_len2 = pose_time.shape[0]
    img_short = 1
else:
    it_len1 = pose_time.shape[0]
    long_time = image_time
    short_time = pose_time
    img_short = 0
#it_len = image_time.shape[0]
first = 0
for i in range(0,it_len1):
    cur_ref_time_ns = np.round(short_time[i,0]/(10**8),1)
    cur_ref_time_ms = np.round(short_time[i,1]/(10**(-7)),2)
    cur_ref_time = [cur_ref_time_ns,cur_ref_time_ms]
    for j in range(0,it_len2):
        odom_time_ns = np.round(long_time[j,0]/(10**8),1)
        odom_time_ms = np.round(long_time[j,1]/(10**(-7)),2)
        odom_time = [odom_time_ns, odom_time_ms]
        if cur_ref_time == odom_time:
            print(i, it_len1-1)
            #time.sleep(2)
            hdf5_data["pose_time"].append(long_time[j,:])
            hdf5_data["image_time"].append(short_time[i,:])
            if img_short == 1:
                hdf5_data["observation"].append(imgs[i,:,:,:])
                hdf5_data["position"].append(pos[j,:])
                hdf5_data["orientation"].append(quat[j,:])
                if first == 0:
                    pos_ref = pos[j,:]
                    first += 1
                rel_pos = pos[j,:] - pos_ref
                hdf5_data["relative_position"].append(rel_pos)
            else:
                hdf5_data["observation"].append(imgs[j,:,:,:])
                hdf5_data["position"].append(pos[i,:]) 
                hdf5_data["orientation"].append(quat[i,:]) 
                if first == 0:
                    pos_ref = pos[i,:]
                    first += 1
                rel_pos = pos[i,:] - pos_ref
                hdf5_data["relative_position"].append(rel_pos)

print('Items in group dict',list(hdf5_data.keys()))
#dump the new dict in new hdf5 file
def dump(output_dir,hdf5_data):
        print('stored data in',output_dir)
        output_hdf5_path = output_dir + '/data4_preproc2' + '.hdf5'
        hdf5_file = h5py.File(output_hdf5_path, "a")
        episode_group = hdf5_file.create_group("preproc_data")
        for sensor_name in hdf5_data.keys():
            episode_group.create_dataset(
                sensor_name, data=np.stack(hdf5_data[sensor_name])
            )
        hdf5_file.close() 

dump(current_dir,hdf5_data)
#delete the previous dataset
# os.remove("/home/gaetan/data/hdf5/test_sim_flight/data3.hdf5")
# print("File Removed!")