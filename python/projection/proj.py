import numpy as np
import transform_mat

def corner_proj(imgs,quat,pos,quat_down_link,pos_down_link,quat_down_optical_frame,pos_down_optical_frame):
    #calculating static transformation matrices on drone itself
    T_baselink_downlink = transform_mat.transf_mat(quat_down_link[0,:],pos_down_link[0,:])
    T_downlink_downoptframe = transform_mat.transf_mat(quat_down_optical_frame[0,:],pos_down_optical_frame[0,:])
    T_baselink_downlink_inv = np.linalg.inv(T_baselink_downlink)
    T_downlink_downoptframe_inv = np.linalg.inv(T_downlink_downoptframe)
    T_baselink_downoptframe = np.matmul(T_baselink_downlink,T_downlink_downoptframe)
    p_downoptframe_baselink = T_baselink_downoptframe[:,3]
    # define the world corner logo positions 
    #for the current dataset
    abs_y_max = 0.68/2
    abs_x_max =0.98/2
    z = 0.001
    c_world = np.array([[abs_x_max,-abs_x_max,-abs_x_max,abs_x_max],
                        [-abs_y_max,-abs_y_max,abs_y_max,abs_y_max],
                        [z,z,z,z],
                        [1,1,1,1]])
    c_size = c_world.shape
    img_size = imgs.shape
    horizontal_field_of_view = (80 * img_size[2]/img_size[1]) * 3.14 / 180
    vertical_field_of_view = 80 * 3.14 / 180
    fx = -img_size[2]/2*np.tan(horizontal_field_of_view/2)**(-1)
    fy = -img_size[1]/2*np.tan(vertical_field_of_view/2)**(-1)
    T_world_baselink = transform_mat.transf_mat(quat,pos)
    p_downoptframe_world  = np.matmul(T_world_baselink,p_downoptframe_baselink)
    T_world_baselink_inv = np.linalg.inv(T_world_baselink)
    c_baselink = np.matmul(T_world_baselink_inv,c_world)
    c_downlink = np.matmul(T_baselink_downlink_inv,c_baselink)
    c_cam = np.matmul(T_downlink_downoptframe_inv,c_downlink)
    K = np.array([[-fx,0,img_size[2]/2],
                 [0,-fy,img_size[1]/2],
                 [0,0,1]])
    for j in range(0,c_size[1]):
        c_cam[0:3,j]=c_cam[0:3,j]/c_cam[2,j]
    c_proj = np.matmul(K,c_cam[0:3,:])
    return c_proj,p_downoptframe_world