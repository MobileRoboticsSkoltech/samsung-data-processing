import cv2
import numpy as np
import open3d as o3d


def __project_pc_to_depth(pcd, new_camera, scaling_factor=5000):
    shape = new_camera.shape
    I = np.zeros(shape, np.float32)
        
    points = np.asarray(pcd.points)
    d = np.linalg.norm(points, axis=1)
    normalized_points = points / np.expand_dims(points[:, 2], axis=1)
    proj_pcd = np.round(new_camera.K_undist @ normalized_points.T).astype(np.int)[:2].T
    
    h, w = shape
    proj_mask = (proj_pcd[:, 0] >= 0) & (proj_pcd[:, 0] < w) & (proj_pcd[:, 1] >= 0) & (proj_pcd[:, 1] < h)
    proj_pcd = proj_pcd[proj_mask, :]
    d = d[proj_mask]
    
    pcd_image = np.zeros(new_camera.shape)
    pcd_image[proj_pcd[:, 1], proj_pcd[:, 0]] = d * scaling_factor
    
    return pcd_image


def reproject_depth(depth_undistorted, original_camera, new_camera, T_original2new):
    image = o3d.geometry.Image(depth_undistorted)
    
    o3d_intrinsic = o3d.camera.PinholeCameraIntrinsic()
    o3d_intrinsic.height, o3d_intrinsic.width = original_camera.shape
    o3d_intrinsic.intrinsic_matrix = original_camera.K_undist
    
    pcd = o3d.geometry.PointCloud().create_from_depth_image(image, o3d_intrinsic)
    pcd = pcd.transform(T_original2new)
    
    reprojected_depth = __project_pc_to_depth(pcd, new_camera)
    
    return reprojected_depth.astype(np.uint16)

def undistort_image(image, camera):
    return cv2.remap(image, camera.map_x_undist, camera.map_y_undist, cv2.INTER_NEAREST)
