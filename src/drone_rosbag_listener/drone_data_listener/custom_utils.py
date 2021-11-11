#!/usr/bin/python3.8
from cv_bridge import CvBridge
import numpy as np
import skimage.transform as sm

bridge = CvBridge()

def resize_image(img, sensor_stats):
        if 'height' in sensor_stats.keys() and 'width' in sensor_stats.keys():
            if 'depth' in sensor_stats.keys():
                size = [sensor_stats['height'], sensor_stats['width'], 3]
        else:
            return img
        if 'depth' in sensor_stats.keys():
            size[2] = sensor_stats['depth']
        scale = [max(int(img.shape[i] / size[i]), 1) for i in range(2)]
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=-1)
        img = img[::scale[0],
            ::scale[1],
            :]
        img = sm.resize(img, size, mode='constant').astype(np.float32)
        if size[-1] == 1 and img.shape[-1] != 1:
            img = img.mean(axis=-1, keepdims=True)
        return img

def process_image(msg, sensor_stats = None):
# if sensor_stats['depth'] == 1:
#     img = bridge.imgmsg_to_cv2(msg, 'passthrough')
#     max_depth = float(sensor_stats['max_depth']) if 'max_depth' in sensor_stats.keys() else 4
#     min_depth = float(sensor_stats['min_depth']) if 'min_depth' in sensor_stats.keys() else 0.1
#     img = np.clip(img, min_depth, max_depth).astype(np.float32)
#     # TODO add image resize and smoothing option
#     print('WARNING: utils.py: depth image is not resized.')
#     return img
# else:
    img = bridge.imgmsg_to_cv2(msg, msg.encoding)
    return resize_image(img, sensor_stats) if sensor_stats is not None else img