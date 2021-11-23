import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv

def crop_img(logo_temp):
    size = logo_temp.shape
    horiz_bounds = []
    vert_bounds = []
    print(size)
    for j in range(0,size[0]):
        if all(logo_temp[j,1000:3000] < 255) and all(logo_temp[j+1,1000:3000] == 255):
            vert_bounds.append(j)
        elif all(logo_temp[j+1,1000:3000] < 255) and all(logo_temp[j,1000:3000] == 255):
            vert_bounds.append(j)
            break

    for i in range(0,size[1]):
        if all(logo_temp[1000:3000,i] < 255) and all(logo_temp[1000:3000,i+1] == 255):
            horiz_bounds.append(i)
        elif all(logo_temp[1000:3000,i+1] < 255) and all(logo_temp[1000:3000,i] == 255):
            horiz_bounds.append(i)
            break
    cropped_img = logo_temp[vert_bounds[0]:vert_bounds[1],horiz_bounds[0]:horiz_bounds[1]]
    return cropped_img
