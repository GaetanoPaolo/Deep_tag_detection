import numpy as np

#function to transform quaternions in transformation matrix
def transf_mat(q,p):
    qx = q[0]
    qy = q[1]
    qz = q[2]
    qw = q[3]
    a11 = -2*(qy**2+qz**2)+1
    a12 = 2*qx*qy-2*qz*qw
    a13 = 2*qx*qz+2*qy*qw
    a21 = 2*qx*qy+2*qz*qw
    a22 = -2*(qx**2+qz**2)+1
    a23 = 2*(qy*qz-qx*qw)
    a31 = 2*(qx*qz-qy*qw)
    a32 = 2*(qy*qz+qw*qx)
    a33 = (-2*(qx**2+qy**2)+1)
    T = np.matrix([[a11, a12, a13, p[0]],
                      [a21, a22, a23, p[1]],
                      [a31, a32, a33, p[2]],
                      [0,0,0,1]],dtype = np.float32)
    return T