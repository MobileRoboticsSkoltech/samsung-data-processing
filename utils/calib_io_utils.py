import json
import numpy as np

from utils.camera import Camera

def __camera_intrinsics_from_coefs_dict(coefs_dict):
    intrinsics_coefs_dict = coefs_dict['intrinsics']['parameters']['parameters_as_dict']
    K = np.eye(3)
    K[0, 0] = intrinsics_coefs_dict['fx']
    K[1, 1] = intrinsics_coefs_dict['fy']
    K[0, 2] = intrinsics_coefs_dict['cx']
    K[1, 2] = intrinsics_coefs_dict['cy']
    
    dist_coefs = np.asarray([
        intrinsics_coefs_dict['k1'],
        intrinsics_coefs_dict['k2'],
        intrinsics_coefs_dict['p1'],
        intrinsics_coefs_dict['p1'],
        intrinsics_coefs_dict['k3'],
        intrinsics_coefs_dict['k4'],
        intrinsics_coefs_dict['k5'],
        intrinsics_coefs_dict['k6']
    ])
    
    height = coefs_dict['resolution_height']
    width = coefs_dict['resolution_width']
    
    return Camera(K, dist_coefs, height, width)


def load_azure_params(calib_params_path):
    with open(calib_params_path, 'r') as calib_file:
        calib_params = json.load(calib_file)
        
    color_intrinsics_dict = calib_params['color_camera']
    color_camera = __camera_intrinsics_from_coefs_dict(color_intrinsics_dict)
    
    depth_intrinsics_dict = calib_params['depth_camera']
    depth_camera = __camera_intrinsics_from_coefs_dict(depth_intrinsics_dict)
    
    R = np.array(calib_params['color_camera']['extrinsics']['rotation']).reshape(3, 3)
    t = np.array(calib_params['color_camera']['extrinsics']['translation_in_meters'])
    T_color2depth = np.eye(4)
    T_color2depth[:3, :3] = R
    T_color2depth[:3, 3] = t
    
    return color_camera, depth_camera, T_color2depth
