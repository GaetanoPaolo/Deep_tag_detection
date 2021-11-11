from matplotlib import pyplot as plt
import cv2
import os
import numpy as np
import h5py


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
# for i in range(0,pos_size[0]):
#     rel_pos [i,:] = pos[i,:] - pos[0,:]

# define new dict for hdf5 storage
hdf5_data = {"observation": [], "position": [], "orientation": [],"relative_position": [],'image_time': [], 'pose_time': []}

#check which timestamps are equal and synchronize images and poses
if image_time.shape[0] < pose_time.shape[0]:
    it_len = image_time.shape[0]
    long_time = pose_time
    short_time = image_time
    img_short = 1
else:
    it_len = pose_time.shape[0]
    long_time = image_time
    short_time = pose_time
    img_short = 0
#it_len = image_time.shape[0]
first = 0
for i in range(0,it_len):
    # if i < 100:
    #     print(image_time[i,:],'____',pose_time[i,:])
        #print(np.round(image_time[i,0]/(10**8),1))
    cur_ref_time = np.round(short_time[i,0]/(10**8),1)
    count = -1
    if cur_ref_time > 1.0:
        odom_time = np.round(long_time[i,0]/(10**8),1)
        # if i < 10:
        #     print(cur_ref_time,'____',odom_time)
        while cur_ref_time != odom_time and i+count < pos_size[0]-1:
            # if i < 10:
            #     print(cur_ref_time,'____',odom_time)
            count += 1
            odom_time = np.round(long_time[i+count,0]/(10**8),1)


        hdf5_data["pose_time"].append(long_time[i+count,:])
        hdf5_data["image_time"].append(short_time[i,:])
        if img_short == 1:
            hdf5_data["observation"].append(imgs[i,:,:,:])
            hdf5_data["position"].append(pos[i+count,:])
            hdf5_data["orientation"].append(quat[i+count,:])
            if first == 0:
                pos_ref = pos[i+count,:]
                first += 1
            rel_pos = pos[i+count,:] - pos_ref
            hdf5_data["relative_position"].append(rel_pos)
        else:
            hdf5_data["observation"].append(imgs[i+count,:,:,:])
            hdf5_data["position"].append(pos[i,:]) 
            hdf5_data["orientation"].append(quat[i,:]) 
            if first == 0:
                pos_ref = pos[i,:]
                first += 1
            rel_pos = pos[i,:] - pos_ref
            hdf5_data["relative_position"].append(rel_pos)
        
        # hdf5_data["relative_position"].append(rel_pos[i,:,])
print('Items in group dict',list(hdf5_data.keys()))
#dump the new dict in new hdf5 file
def dump(output_dir,hdf5_data):
        print('stored data in',output_dir)
        output_hdf5_path = output_dir + '/data4_preproc' + '.hdf5'
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