"""
2024.05, Sogang Univ. Jeongkyun Park

Lip Occlusion for multi-view augmentation and evaluation
"""

import os
import json
import argparse
from typing import List, Union

import cv2
import numpy as np
from scipy.stats import beta


def capture_video(video_path, end_idx=None):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    
    ret = True
    frame_idx = -1
    frames = []
    while ret:
        ret, img = cap.read()
        if ret:
            frame_idx += 1
            if end_idx is not None and frame_idx == end_idx:
                break
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            frames.append(img)
    cap.release()
    frames = np.stack(frames)
    return frames, fps, width, height

def write_video(save_path, video_array, fps, width, height):
    # Initialize the VideoWriter object for the first segment
    out = cv2.VideoWriter(save_path,
                          cv2.VideoWriter_fourcc(*'mp4v'),  # Use 'avc1' if 'mp4v' doesn't work
                          fps, (width, height))
    for frame in video_array:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out.write(frame)
    out.release()

def project_occluder_on_image(obj, obj_coords, image):
    # Assuming `image` is of shape (H, W, C) and `obj` is of shape (h, w) or (h, w, c)
    bg_h, bg_w, bg_c = image.shape
    if obj.ndim == 2:
        obj_h, obj_w = obj.shape
    elif obj.ndim == 3:
        obj_h, obj_w, _ = obj.shape
    
    obj_hc, obj_wc = map(int, obj_coords)  # Object center coordinates
    
    # Image slice start and end
    bg_hs_ = obj_hc - obj_h // 2
    bg_he = max(0, min(bg_h, bg_hs_ + obj_h))
    bg_hs = min(max(0, bg_hs_), bg_h)
    
    bg_ws_ = obj_wc - obj_w // 2
    bg_we = max(0, min(bg_w, bg_ws_ + obj_w))
    bg_ws = min(max(0, bg_ws_), bg_w)
    
    # Object slice start and end
    obj_hs = max(0, - bg_hs_)
    obj_he = obj_hs + bg_he - bg_hs
    obj_ws = max(0, - bg_ws_)
    obj_we = obj_ws + bg_we - bg_ws
    
    # Project an object onto image
    if obj.ndim == 2:
        image[bg_hs:bg_he, bg_ws:bg_we] = obj[obj_hs:obj_he, obj_ws:obj_we, None]
    elif obj.ndim == 3:
        image[bg_hs:bg_he, bg_ws:bg_we] = obj[obj_hs:obj_he, obj_ws:obj_we]

def rotation_matrix(rtype:str, theta:float):
    theta = theta / 180 * np.pi
    if rtype == "y":
        mat = np.array([[1, 0, 0],
                        [0, np.cos(theta), -np.sin(theta)],
                        [0, np.sin(theta),  np.cos(theta)]])
    elif rtype == "p":
        mat = np.array([[np.cos(theta), 0, np.sin(theta)],
                        [0, 1, 0],
                        [-np.sin(theta), 0, np.cos(theta)]])
    elif rtype == "r":
        mat = np.array([[np.cos(theta), -np.sin(theta), 0],
                        [np.sin(theta),  np.cos(theta), 0],
                        [0, 0, 1]])
    else:
        mat = np.array([[1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1]])
    return mat

def bound_point(point, min_boundary, max_boundary):
    min_exceed = point < min_boundary
    max_exceed = point > max_boundary
    point = np.max(np.stack([point, min_boundary]), axis=0) # Min boundary
    point = np.min(np.stack([point, max_boundary]), axis=0) # Min boundary
    return point, min_exceed, max_exceed

class Camera:
    def __init__(
        self, 
        principal_points:List[float] = [0,0,0],
        rotation_degree:float = 0, rot_type:str = 'n', 
        focal_length:float = 2.86, sensor_size:List[float] = [3.47,6.17],
        projected_image_shape:List[int] = [1080, 1920],
        cropped_img_shape:List[int] = [96, 96],
    ):
        # World system
        self.f_world = focal_length
        self.s_world = sensor_size
        
        # Pixel system
        self.f_pixel = self.f_world * projected_image_shape[0] / sensor_size[0]
        self.s_pixel = projected_image_shape
        
        self.cropped_img_shape = cropped_img_shape
        self.projected_image_shape = projected_image_shape
        
        # For camera geometry
        self.pp = principal_points
        self.rot_mat = rotation_matrix(rot_type, rotation_degree)
        self.rot_mat_inv = np.linalg.inv(self.rot_mat)
        
        # Lip ROIs in memory
        self._roi_frames = None
        self._max_boundary = None
    
    def set_frame_roi(self, roi):
        self._roi_frames = roi
        self._max_boundary = np.array([*roi[:,:2].min(axis=0), *roi[:,2:].max(axis=0)])
        # print(self._max_boundary)
    
    def get_frame_roi(self, roi, frame_idx):
        return self._roi_frames[frame_idx]
    
    def reset_frame_roi(self):
        self._roi_frames = None
        self._max_boundary = None
    
    def __len__(self):
        return len(self._roi_frames)
    
    def project_from_world_coordinates(self, xyz_array):
        if not isinstance(xyz_array, np.ndarray):
            xyz_array = np.array(xyz_array)
        xyz_out = (xyz_array - self.pp) @ self.rot_mat
        return xyz_out
    
    def project_to_world_coordinates(self, xyz_array):
        if not isinstance(xyz_array, np.ndarray):
            xyz_array = np.array(xyz_array)
        xyz_out = xyz_array @ self.rot_mat_inv + self.pp
        return xyz_out
    
    def project_to_img_from_cam_coordinates(self, xyz_array, frame_idx:int=None):
        # Project (x, y, z) to the 2D space
        x, y, z = xyz_array
        x_img, y_img, _ = xyz_array * self.f_pixel / z
        # Uncentering
        x_img += self.projected_image_shape[0]//2
        y_img += self.projected_image_shape[1]//2
        # Crop lip boundary
        # if self._roi_frames is not None:
        #     xmin, ymin, xmax, ymax = self._roi_frames[frame_idx]
        #     x_scale = self.cropped_img_shape[0] / (xmax - xmin)
        #     y_scale = self.cropped_img_shape[1] / (ymax - ymin)
        #     x_img = int((x_img - xmin) * x_scale)
        #     y_img = int((y_img - ymin) * y_scale)
        return x_img, y_img
        
    def overlap_occluder_on_img(self, occluder, occluder_proj, occluder_depth, image):
        # Overlap
        x_scale = min(self.f_pixel / occluder_depth, self.cropped_img_shape[0] / occluder.shape[0])
        y_scale = min(self.f_pixel / occluder_depth, self.cropped_img_shape[1] / occluder.shape[1])
        obj_resized = cv2.resize(occluder, dsize=(0,0), fx=x_scale, fy=y_scale)
        project_occluder_on_image(obj_resized, occluder_proj, image)
    
    def generate_occluder(
        self, 
        occluder_type:str = "box", max_distance:float = 500, min_distance:float = 10,
        max_shape:Union[float, List[float]] = [50, 50], min_shape:Union[float, List[float]] = [10, 10]
    ):
        # Generate object
        if occluder_type == "box":
            min_size = min_shape if isinstance(min_shape, float) else min_shape[0]
            max_size = max_shape if isinstance(max_shape, float) else max_shape[0]
            obj_shape = int(np.random.uniform(min_size, max_size))
            obj = np.zeros((obj_shape, obj_shape))
        else:
            raise NotImplementedError
        
        # Select object distance from random uniform distribution
        xtl, ytl, xbr, ybr = self._max_boundary
        xtl_cent, xbr_cent = map(lambda x: x-self.s_pixel[0]/2, (xtl, xbr))
        ytl_cent, ybr_cent = map(lambda y: y-self.s_pixel[1]/2, (ytl, ybr))
        obj_z_cam = np.random.uniform(min_distance, max_distance)
        # # Select object xy-coordinates from random uniform distribution
        # obj_x_img = np.random.uniform(xtl_cent, xbr_cent)
        # obj_y_img = np.random.uniform(ytl_cent, ytl_cent)
        # Select object xy-coordinates from beta distribution (a=5, b=5)
        alpha, bet = 5, 5
        obj_x_img = xtl_cent + beta.rvs(alpha, bet) * (xbr_cent - xtl_cent)
        obj_y_img = ytl_cent + beta.rvs(alpha, bet) * (ybr_cent - ytl_cent)
        # # Select object xy-coordinates from deterministic center
        # obj_x_img = xtl_cent + (xbr_cent - xtl_cent) / 2
        # obj_y_img = ytl_cent + (ybr_cent - ytl_cent) / 2
        # Normalize
        obj_x_cam = obj_x_img * obj_z_cam / self.f_pixel
        obj_y_cam = obj_y_img * obj_z_cam / self.f_pixel
        
        print(f"Generated occluder position {tuple(map(int,(obj_x_cam, obj_y_cam, obj_z_cam)))}")
        
        return obj, (obj_x_cam, obj_y_cam, obj_z_cam)
    
    def move_occluder(self, occluder_coords, max_t:int = 100, zmin:float = 1e-4, zmax:float = 500):
        occluder_coords = occluder_coords if isinstance(occluder_coords, np.ndarray) else np.array(occluder_coords)
        # Initialize parameters for defining movement
        vector = np.random.uniform(-2, 2, size=3)
        accel = np.array([0., 0, 0])
        # Define boundary
        min_bound_centered = self._max_boundary[:2] - np.array([self.projected_image_shape[0]/2, self.projected_image_shape[1]/2])
        max_bound_centered = self._max_boundary[2:] - np.array([self.projected_image_shape[0]/2, self.projected_image_shape[1]/2])
        # Iterate
        for t in range(max_t):
            # Get xyz-boundary on current z-position
            min_boundary = np.array([*min_bound_centered * occluder_coords[-1] / self.f_pixel, zmin])
            max_boundary = np.array([*max_bound_centered * occluder_coords[-1] / self.f_pixel, zmax])
            # Accumulate randomness to the vector
            dv = 0.1 * np.random.randn(3)
            vector += dv
            # Move the point
            occluder_coords = occluder_coords + vector + accel * t # accleration using 1st derivative
            # Adjust points outbounded
            occluder_coords, min_ex, max_ex = bound_point(occluder_coords, min_boundary, max_boundary)
            vector[min_ex | max_ex] *= -1 # Invert the direction
            yield occluder_coords

def main(args):
    height = 1080
    width = 1920
    height_crop = 96
    width_crop = 96

    # 1. Initialize camera instances
    cameras = []
    for i, (rot, rtype) in enumerate(zip(args.camera_rot, args.camera_rot_type)):
        pp = np.array(args.camera_principal_point[3*i:3*i+3])
        camera = Camera(pp, rot, rtype, args.focal_length, args.sensor_size, [height, width])
        cameras.append(camera)
    
    # 2. Load target video and boundaries
    video_lst = []
    for camera, input, boundary_path in zip(cameras, args.input, args.boundary_path):
        video, fps, _, _ = capture_video(input, 100)
        video_lst.append(video)
        
        boundaries = np.load(boundary_path)
        camera.set_frame_roi(boundaries)
    target_camera = cameras[args.target_idx]
    
    # 3. Initialize occluder's camera-system coordinates
    obj, obj_coords = target_camera.generate_occluder(
        occluder_type=args.object_type, max_shape = args.object_size, min_distance=args.target_distance-100, max_distance=args.target_distance,
    )

    # 4. Simulate the movement of the occluder
    movement_generator = target_camera.move_occluder(obj_coords, len(video), zmin=1e-4, zmax=args.target_distance)

    # 5. Project occluder on the original videos
    for frame_idx, xyzt in enumerate(movement_generator):
        # Map (xt, yt, zt) to the target camera coordinates
        xyzt = cameras[args.target_idx].project_to_world_coordinates(xyzt)
        for i, (camera, video) in enumerate(zip(cameras, video_lst)):
            # Shift and rotate (xt, yt, zt) for each view
            xyzt_ = camera.project_from_world_coordinates(xyzt)
            if xyzt_[-1] < 0:
                continue
            # Project (xt, yt, zt) to the 2D image space
            x_img, y_img = camera.project_to_img_from_cam_coordinates(xyzt_, frame_idx)
            # Occlude frame
            camera.overlap_occluder_on_img(obj, [x_img, y_img], xyzt_[-1], video[frame_idx])

    print("Start saving the output videos..")
    # 7. Save occluded full video
    for output_path, video in zip(args.output, video_lst):
        # write_video(output_path, video, fps, width_crop, height_crop)
        write_video(output_path, video, fps, width, height)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', nargs='+', help='')
    parser.add_argument('-o', '--output', nargs='+', help='')
    parser.add_argument('-ot', '--object_type', default='box', help='')
    parser.add_argument('-op', '--object_path', required=False, help='')
    parser.add_argument('-os', '--object_size', type=float, help='Occluder maximum size (mm)')
    parser.add_argument('-bp', '--boundary_path', nargs='+', required=False, help='Lip roi filepath')
    
    parser.add_argument('-f', '--focal_length', default=2.86, help='Physical focal length of the target camera (mm)')
    parser.add_argument('-ss', '--sensor_size', nargs=2, default=[3.47, 6.17], help='Physical sensor size (H x W) (mm)')
    parser.add_argument('-d', '--target_distance', default=500, help='Distance between the camera and the target for occlusion (mm)')
    
    parser.add_argument('-cp', '--camera_principal_point', nargs='+', type=int, help='Camera positions in world coordinates (x, y, z) (mm)')
    parser.add_argument('-cr', '--camera_rot', nargs='+', type=float, help='Camera rotation degree')
    parser.add_argument('-crt', '--camera_rot_type', nargs='+', help='Camera rotation type [pitch, yaw, roll]')
    parser.add_argument('-t', '--target_idx', type=int, help='Target view index for occlusion')
    
    args = parser.parse_args()
    
    return args
    

if __name__ == "__main__":
    args = parse_args()
    main(args)