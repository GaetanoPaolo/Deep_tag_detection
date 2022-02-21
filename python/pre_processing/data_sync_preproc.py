from matplotlib import pyplot as plt
import cv2 as cv
import os
import numpy as np
import h5py
import time
import proj
import det_logo_image

# load all the necessary data from the hdf5 file
#f = h5py.File('/home/gaetan/data/hdf5/test_sim_flight/data3.hdf5', 'r')
current_dir = '/home/gaetan/data/hdf5/altitude_test/'
print(current_dir)
f = h5py.File(current_dir+'data4.hdf5', 'r')
base_items = list(f.items())
#print('Groups:',base_items)
dset = f.get('0')
SIM = True
print('Items in group 0',list(dset.items()))
imgs = np.array(dset.get('observation'))
image_time = np.array(dset.get('image_time'))





# define new dict for hdf5 storage
if SIM:
    pose_time = np.array(dset.get('pose_time'))
    pos = np.array(dset.get('position'))
    quat = np.array(dset.get('orientation'))
    pos_base_footprint = np.array(dset.get("pos_base_footprint"))
    quat_base_footprint = np.array(dset.get("quat_base_footprint"))
    pos_base_stabilized = np.array(dset.get("pos_base_stabilized"))
    quat_base_stabilized = np.array(dset.get("quat_base_stabilized"))
    pos_base_link = np.array(dset.get("pos_base_link"))
    quat_base_link = np.array(dset.get("quat_base_link"))
    pos_down_link = np.array(dset.get("pos_down_link"))
    quat_down_link = np.array(dset.get("quat_down_link"))
    pos_down_optical_frame = np.array(dset.get("pos_down_optical_frame"))
    quat_down_optical_frame = np.array(dset.get("quat_down_optical_frame"))
    hdf5_data = {"observation": [], "position": [],"orientation": [],"corners":[],"pos_origin_cam":[],"pos_base_footprint": [],"quat_base_footprint": [], 
            'pos_base_stabilized':[],'quat_base_stabilized':[],'pos_base_link':[],'quat_base_link':[],
            'pos_down_link':[],'quat_down_link':[],'pos_down_optical_frame':[],'quat_down_optical_frame':[],
            'image_time': [], 'pose_time': []}
    hdf5_data['pos_down_link'].append(pos_down_link[0,:])
    hdf5_data['quat_down_link'].append(quat_down_link[0,:])
    hdf5_data['pos_down_optical_frame'].append(pos_down_optical_frame[0,:])
    hdf5_data['quat_down_optical_frame'].append(quat_down_optical_frame[0,:])
else:
    pose_time = np.array(dset.get('est_time'))
    pos = np.array(dset.get("pos_est"))
    quat = np.array(dset.get("quat_est"))
    #odom_pos = np.array(dset.get("odom_position"))
    #odom_quat = np.array(dset.get("odom_orientation"))
    #odom_time = np.array(dset.get("odom_time"))
    K = np.array(dset.get("K"))
    P = np.array(dset.get("P"))
    hdf5_data = {"observation": [], "position": [],"orientation": [], 'relative_position':[],'image_time': [], 'pose_time': [], 'K':[], 'P':[]}
    hdf5_data['K'].append(K[0,:])
    hdf5_data['P'].append(P[0,:])
#rel_pos = pos
pos_size = pos.shape[0]
print(pose_time.shape)
if pose_time.shape[0] > pos_size:
    pose_time_amount = pos_size
else:
    pose_time_amount = pose_time.shape[0]
#check which timestamps are equal and synchronize images and poses
if image_time.shape[0] < pose_time_amount:
    it_len1 = image_time.shape[0]
    long_time = pose_time
    short_time = image_time
    it_len2 = pose_time_amount
    img_short = 1
else:
    it_len1 = pose_time_amount
    long_time = image_time
    short_time = pose_time
    img_short = 0
    it_len2 = image_time.shape[0]
first = 0
print(img_short)
def dump(output_dir,hdf5_data,ep):
        print('stored data in',output_dir)
        output_hdf5_path = output_dir + '/data4_sync' + '.hdf5'
        hdf5_file = h5py.File(output_hdf5_path, "a")
        episode_group = hdf5_file.create_group(str(ep))
        for sensor_name in hdf5_data.keys():
            print(sensor_name)
            episode_group.create_dataset(
                sensor_name, data=np.stack(hdf5_data[sensor_name])
            )
        hdf5_file.close()
count = 0
ep = 1
for i in range(0,it_len1):
    #cur_ref_time_ns = np.round(short_time[i,0]/(10**8),1)
    cur_ref_time_ns = np.floor(short_time[i,0]/(10**7))
    cur_ref_time_ms = np.round(short_time[i,1]/(10**(-7)),2)
    cur_ref_time = [cur_ref_time_ns,cur_ref_time_ms]
    for j in range(0,it_len2):
        #odom_time_ns = np.round(long_time[j,0]/(10**8),1)
        odom_time_ns = np.floor(long_time[j,0]/(10**7))
        odom_time_ms = np.round(long_time[j,1]/(10**(-7)),2)
        odom_time = [odom_time_ns, odom_time_ms]
        # print(cur_ref_time),
        # print(odom_time)
        # print(cur_ref_time == odom_time)
        # print('__________________________')
        # time.sleep(0.5)
        if cur_ref_time == odom_time:
            count += 1
            hdf5_data["image_time"].append(long_time[j,:])
            #hdf5_data["odom_time"].append(odom_time[j,:])
            hdf5_data["pose_time"].append(short_time[i,:])
            if img_short == 1:
                print ('i'),
                print(i)
                if det_logo_image.logo_image(SIM,imgs[i,:,:,:]):
                    hdf5_data["observation"].append(imgs[i,:,:,:])
                    hdf5_data["position"].append(pos[j,:])
                    hdf5_data["orientation"].append(quat[j,:])
                    if SIM:
                        
                        hdf5_data["pos_base_footprint"].append(pos_base_footprint[j,:])
                        hdf5_data["quat_base_footprint"].append(quat_base_footprint[j,:])
                        hdf5_data['pos_base_stabilized'].append(pos_base_stabilized[j,:])
                        hdf5_data['quat_base_stabilized'].append(quat_base_stabilized[j,:])
                        hdf5_data['pos_base_link'].append(pos_base_link[j,:])
                        hdf5_data['quat_base_link'].append(quat_base_link[j,:])
                        corn, p_cam = proj.corner_proj(imgs,quat[j,:], pos[j,:],quat_down_link,pos_down_link,quat_down_optical_frame,pos_down_optical_frame)
                        hdf5_data["corners"].append(np.array(corn))
                        hdf5_data["pos_origin_cam"].append(np.array(p_cam))
                    else:
                        #hdf5_data["odom_position"].append(odom_pos[j,:])
                        #hdf5_data["odom_orientation"].append(odom_quat[j,:])
                        if first == 0:
                            pos_ref = pos[j,:]
                            first += 1
                        rel_pos = pos[j,:] - pos_ref
                        hdf5_data["relative_position"].append(rel_pos)
            else: 
                print('j'),
                print(j)
                if det_logo_image.logo_image(SIM,imgs[j,:,:,:]):
                    hdf5_data["observation"].append(imgs[j,:,:,:])
                    hdf5_data["position"].append(pos[i,:]) 
                    hdf5_data["orientation"].append(quat[i,:]) 
                    if SIM:
                        # std_corners = [[0,0],[0,0],[0,0],[0,0]]
                        # corn = cd_harris.cornerHarris(140,imgs[j,:,:,:]*255)
                        # for t in range(0,len(corn)):
                        #     std_corners[t] = corn[t]
                        #print(len(std_corners))
                        hdf5_data["pos_base_footprint"].append(pos_base_footprint[i,:])
                        hdf5_data["quat_base_footprint"].append(quat_base_footprint[i,:])
                        hdf5_data['pos_base_stabilized'].append(pos_base_stabilized[i,:])
                        hdf5_data['quat_base_stabilized'].append(quat_base_stabilized[i,:])
                        hdf5_data['pos_base_link'].append(pos_base_link[i,:])
                        hdf5_data['quat_base_link'].append(quat_base_link[i,:])
                        corn, p_cam = proj.corner_proj(imgs,quat[i,:], pos[i,:],quat_down_link,pos_down_link,quat_down_optical_frame,pos_down_optical_frame)
                        hdf5_data["corners"].append(np.array(corn))
                        hdf5_data["pos_origin_cam"].append(np.array(p_cam))
                    else:
                        #hdf5_data["odom_position"].append(odom_pos[i,:])
                        #hdf5_data["odom_orientation"].append(odom_quat[i,:])
                        if first == 0:
                            pos_ref = pos[i,:]
                            first += 1
                        rel_pos = pos[i,:] - pos_ref
                        hdf5_data["relative_position"].append(rel_pos)
            break
    if count >= 300:
        dump(current_dir,hdf5_data,ep)
        count = 0
        ep += 1
        if SIM:
            hdf5_data = {"observation": [], "position": [],"orientation": [],"corners":[],"pos_origin_cam":[],"pos_base_footprint": [],"quat_base_footprint": [], 
            'pos_base_stabilized':[],'quat_base_stabilized':[],'pos_base_link':[],'quat_base_link':[],
            'pos_down_link':[],'quat_down_link':[],'pos_down_optical_frame':[],'quat_down_optical_frame':[],
            'image_time': [], 'pose_time': []}
            hdf5_data['pos_down_link'].append(pos_down_link[0,:])
            hdf5_data['quat_down_link'].append(quat_down_link[0,:])
            hdf5_data['pos_down_optical_frame'].append(pos_down_optical_frame[0,:])
            hdf5_data['quat_down_optical_frame'].append(quat_down_optical_frame[0,:])
        else:
            hdf5_data = {"observation": [], "position": [],"orientation": [], 'relative_position':[],'image_time': [], 'pose_time': []}

print('Items in group dict',list(hdf5_data.keys()))
#dump the new dict in new hdf5 file
 


