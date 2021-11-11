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
from std_msgs.msg import String, Float32MultiArray, Empty
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from rosgraph_msgs.msg import Clock
import custom_utils

SIM = False

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
            rospy.Subscriber('/ground_truth_to_tf/pose',
                                PoseStamped,
                                callback=self._pose_callback)
            self.size = {'height': 200, 'width': 200, 'depth': 3}
        else:
            rospy.Subscriber('/camera/fisheye1/image_raw',
                                Image,
                                callback=self._camera_callback)
            rospy.Subscriber('/mavros/local_position/odom',
                                Odometry,
                                callback=self._odom_callback)
            self.size = {'height': 480, 'width': 640, 'depth': 3}
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
        self._hdf5_data = {"observation": [], "position": [],"orientation": [], 'image_time': [], 'pose_time': []}
    
    def _camera_callback(self, msg):
        h = getattr(msg, 'header')
        stamp = h.stamp
        self._hdf5_data["image_time"].append([stamp.nsecs, stamp.secs*1e-9])
        image = custom_utils.process_image(msg, self.size)
        self._hdf5_data["observation"].append(deepcopy(image))
    def _odom_callback(self, msg):
        h = getattr(msg, 'header')
        stamp = h.stamp
        self._hdf5_data["pose_time"].append([stamp.nsecs, stamp.secs*1e-9])
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
        self._hdf5_data["position"].append(pos_vect)
        self._hdf5_data["orientation"].append(orientation_vect)
    def _pose_callback(self,msg):
        h = getattr(msg, 'header')
        stamp = h.stamp
        self._hdf5_data["pose_time"].append([stamp.nsecs, stamp.secs*1e-9])
        p = getattr(msg, 'pose')
        position = p.position
        quaternion = p.orientation
        pos_vect = [position.x, position.y, position.z]
        orientation_vect = [quaternion.x, quaternion.y, quaternion.z, quaternion.w]
        self._hdf5_data["position"].append(pos_vect)
        self._hdf5_data["orientation"].append(orientation_vect)
    def run(self):
        rate = rospy.Rate(100)
        while not rospy.is_shutdown() and len(rospy.get_published_topics()) > 2:
            rate.sleep()

print(__name__)
if __name__ == '__main__':
    print('protocol started')
    output_directory = '/home/gaetan/data/hdf5/landing_T265_001'
    data_saver = Datasaver(output_dir=output_directory)
    print('Datasaver_created')
    data_saver.run()
    data_saver._dump()
