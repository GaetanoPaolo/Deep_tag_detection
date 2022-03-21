import numpy as np
#This function detects whether the psi logo is present in the image 
#by detecting the presence of  non - green pixels
def logo_image(SIM,img):
    if SIM:
        R_slice = np.sum(img[:,:,:], axis = 2)
        size = R_slice.shape
        template = np.ones((size[0],size[1]))*2.1058824
        diff = R_slice - template
        #print(np.sum(np.sum(diff,0),0))
        if np.sum(np.sum(diff,0),0) < 0.1:
            return False
        else:
            return True
    else:
        return True