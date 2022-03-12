#!/usr/bin/env python
import os
from sys import argv
import shutil
import time
import unittest
from copy import deepcopy

import h5py
from std_msgs.msg import Empty
import numpy as np
import rospy
from geometry_msgs.msg import Twist, PointStamped, Point, PoseStamped
from tf2_msgs.msg import TFMessage
from std_msgs.msg import String, Float32MultiArray, Empty
from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import Odometry
from rosgraph_msgs.msg import Clock
import custom_utils

SIM = True

class Datasaver:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        rospy.init_node('datasaver')
        # wait till ROS started properly
        stime = time.time()
        max_duration = 60
        while not rospy.has_param('/output_path') and time.time() < stime + max_duration:
            time.sleep(0.01)
        if SIM:
            rospy.Subscriber('/down/image_raw', 
                                Image,
                                callback=self._camera_callback)
            rospy.Subscriber('/ground_truth/state',
                                Odometry,
                                callback=self._pose_callback)
            rospy.Subscriber('/tf', TFMessage,callback=self._tf_callback)
            rospy.Subscriber('/tf_static', TFMessage,callback=self._tfstatic_callback)
            self.size = {'height': 200, 'width': 200, 'depth': 3}
        else:
            rospy.Subscriber('/camera/fisheye1/image_raw',
                                Image,
                                callback=self._camera_callback)
            # rospy.Subscriber('/tf',
            #                     TFMessage,
            #                     callback=self._tf_callback_d400)
            rospy.Subscriber('/camera/fisheye1/camera_info',
                                CameraInfo,
                                callback=self._camera_info_callback
                                )
            # rospy.Subscriber('/mavros/local_position/odom',
            #                     Odometry,
            #                     callback=self._odom_callback)
            rospy.Subscriber('/mavros/local_position/pose',
                                PoseStamped,
                                callback = self._mavrospose_callback)
            rospy.Subscriber('/drone_pose_estimate',
                                PoseStamped,
                                callback = self._estpose_callback)
            #self.size = {'height': 480, 'width': 640, 'depth': 3}
            self.size = {'height': 800, 'width': 848, 'depth': 1}
        self._episode_id = -1
        self._reset()

    def _dump(self):
        print('stored movie in ~/.ros')
        output_hdf5_path = self.output_dir + '/data4' + '.hdf5'
        hdf5_file = h5py.File(output_hdf5_path, "a")
        episode_group = hdf5_file.create_group(str(self._episode_id))
        for sensor_name in self._hdf5_data.keys():
            episode_group.create_dataset(
                sensor_name, data=np.stack(self._hdf5_data[sensor_name])
            )
        hdf5_file.close()        

    def _reset(self):
        self._episode_id += 1
        if SIM:
            self._hdf5_data = {"observation": [], "position": [],"orientation": [],"pos_base_footprint": [],"quat_base_footprint": [], 
            'pos_base_stabilized':[],'quat_base_stabilized':[],'pos_base_link':[],'quat_base_link':[],
            'pos_down_link':[],'quat_down_link':[],'pos_down_optical_frame':[],'quat_down_optical_frame':[],
            'image_time': [], 'pose_time': []}
        else:
            #self._hdf5_data = {"observation": [], "pos_camera_pose_frame": [],"quat_camera_pose_frame": [], "pos_/d400_link":[],"quat_/d400_link":[],'odom_position':[],'odom_orientation':[],'image_time': [], 'pose_time': [], 'odom_time':[],'K':[], 'P':[]}
            self._hdf5_data = {"observation": [], "pos_mavros": [],"pos_est": [], "quat_mavros":[],"quat_est":[],"mavros_time":[],"est_time":[],'image_time': [],'K':[], 'P':[]}
    
    def _camera_callback(self, msg):
        h = getattr(msg, 'header')
        stamp = h.stamp
        self._hdf5_data["image_time"].append([stamp.nsecs, stamp.secs*1e-9])
        image = custom_utils.process_image(msg, self.size)
        self._hdf5_data["observation"].append(deepcopy(image))
    def _camera_info_callback(self,msg):
        self._hdf5_data["K"].append(msg.K)
        self._hdf5_data["P"].append(msg.P)
    def _odom_callback(self, msg):
        h = getattr(msg, 'header')
        stamp = h.stamp
        self._hdf5_data["odom_time"].append([stamp.nsecs, stamp.secs*1e-9])
        p = getattr(msg, 'pose')
        pose = p.pose
        position = pose.position
        quaternion = pose.orientation
        pose_cov = p.covariance
        t = getattr(msg, 'twist')
        twist = t.twist
        linear = twist.linear
        angular = twist.angular
        twist_cov = t.covariance
        pos_vect = [position.x, position.y, position.z]
        orientation_vect = [quaternion.x, quaternion.y, quaternion.z, quaternion.w]
        self._hdf5_data["odom_position"].append(pos_vect)
        self._hdf5_data["odom_orientation"].append(orientation_vect)
    def _pose_callback(self,msg):
        h = getattr(msg, 'header')
        stamp = h.stamp
        self._hdf5_data["pose_time"].append([stamp.nsecs, stamp.secs*1e-9])
        p = getattr(msg, 'pose')
        pose = p.pose
        position = pose.position
        quaternion = pose.orientation
        pos_vect = [position.x, position.y, position.z]
        orientation_vect = [quaternion.x, quaternion.y, quaternion.z, quaternion.w]
        self._hdf5_data["position"].append(pos_vect)
        self._hdf5_data["orientation"].append(orientation_vect)
    def _mavrospose_callback(self,msg):
        h = getattr(msg, 'header')
        stamp = h.stamp
        self._hdf5_data["mavros_time"].append([stamp.nsecs, stamp.secs*1e-9])
        p = getattr(msg, 'pose')
        position = p.position
        quaternion = p.orientation
        pos_vect = [position.x, position.y, position.z]
        orientation_vect = [quaternion.x, quaternion.y, quaternion.z, quaternion.w]
        self._hdf5_data["pos_mavros"].append(pos_vect)
        self._hdf5_data["quat_mavros"].append(orientation_vect)
    def _estpose_callback(self,msg):
        h = getattr(msg, 'header')
        stamp = h.stamp
        self._hdf5_data["est_time"].append([stamp.nsecs, stamp.secs*1e-9])
        p = getattr(msg, 'pose')
        position = p.position
        quaternion = p.orientation
        pos_vect = [position.x, position.y, position.z]
        orientation_vect = [quaternion.x, quaternion.y, quaternion.z, quaternion.w]
        self._hdf5_data["pos_est"].append(pos_vect)
        self._hdf5_data["quat_est"].append(orientation_vect)
    def _tf_callback_d400(self,msg):
        transfs = getattr(msg, 'transforms')
        cur_transf = transfs[0]
        h = cur_transf.header
        stamp = h.stamp
        self._hdf5_data["pose_time"].append([stamp.nsecs, stamp.secs*1e-9])
        cur_pos = cur_transf.transform.translation
        cur_quaternion = cur_transf.transform.rotation
        cur_child_frame = cur_transf.child_frame_id
        self._hdf5_data["pos_"+cur_child_frame].append([cur_pos.x, cur_pos.y, cur_pos.z])
        self._hdf5_data["quat_"+cur_child_frame].append([cur_quaternion.x, cur_quaternion.y, cur_quaternion.z, cur_quaternion.w])
    def _tf_callback(self,msg):
        transfs = getattr(msg, 'transforms')
        for k in range(0,3):
            cur_transf = transfs[k]
            cur_pos = cur_transf.transform.translation
            cur_quaternion = cur_transf.transform.rotation
            cur_child_frame = cur_transf.child_frame_id
            self._hdf5_data["pos_"+cur_child_frame].append([cur_pos.x, cur_pos.y, cur_pos.z])
            self._hdf5_data["quat_"+cur_child_frame].append([cur_quaternion.x, cur_quaternion.y, cur_quaternion.z, cur_quaternion.w])

    def _tfstatic_callback(self,msg):
        transfs = getattr(msg, 'transforms')
        for k in range(0,2):
            cur_transf = transfs[k]
            cur_pos = cur_transf.transform.translation
            cur_quaternion = cur_transf.transform.rotation
            cur_child_frame = cur_transf.child_frame_id
            self._hdf5_data["pos_"+cur_child_frame].append([cur_pos.x, cur_pos.y, cur_pos.z])
            self._hdf5_data["quat_"+cur_child_frame].append([cur_quaternion.x, cur_quaternion.y, cur_quaternion.z, cur_quaternion.w])
    def run(self):
        rate = rospy.Rate(100)
        while not rospy.is_shutdown() and len(rospy.get_published_topics()) > 2:
            rate.sleep()

print(__name__)
if __name__ == '__main__':
    print('protocol started')
    output_directory = '/home/gaetan/data/hdf5/coca_cola_200res_alt_rot'
    data_saver = Datasaver(output_dir=output_directory)
    print('Datasaver_created')
    data_saver.run()
    data_saver._dump()
