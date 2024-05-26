"""
2024.05, Sogang Univ. Jeongkyun Park

Lip Occlusion for multi-view augmentation and evaluation
"""

import os
import json
import argparse

import cv2
import numpy as np


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

# def load_boundary(boundary_path):
#     bbox_dict = {}
#     with open(boundary_path) as f:
#         for line in f:
#             key, xtl, ytl, xbr, ybr = line.split()
#             xtl, ytl, xbr, ybr = map(int, [xtl, ytl, xbr, ybr])
#             bbox_dict[key] = (xtl, ytl, xbr, ybr)
#     return bbox_dict

def load_boundary_from_json(boundary_path):
    # Read ROI
    with open(boundary_path) as f:
        boundaries = json.load(f)
    boundaries = boundaries['Lip_bounding_box']['xtl_ytl_xbr_ybr']
    boundaries = np.array(boundaries)
    # Get maximum boundary
    xtl, ytl = boundaries[:,:2].min(axis=0)
    xbr, ybr = boundaries[:,2:].max(axis=0)
    
    return xtl, ytl, xbr, ybr

def init_object(args, xtl, ytl, xbr, ybr, height, width, f):
    # Generate object
    if args.object_type == "box":
        obj_size = int(np.random.uniform(10, args.object_size))
        obj = np.zeros((obj_size, obj_size))
    else:
        raise NotImplementedError
    
    # Select object position from random uniform distribution
    obj_z_cam = np.random.uniform(10, args.target_distance)
    obj_x_img = np.random.uniform(xtl - height / 2, xbr - height / 2)
    obj_y_img = np.random.uniform(ytl - width / 2, ytl - width / 2)
    # Normalize
    obj_x_cam = obj_x_img * obj_z_cam / f
    obj_y_cam = obj_y_img * obj_z_cam / f
    
    return obj, (obj_x_cam, obj_y_cam, obj_z_cam)

def bound_point(point, min_boundary, max_boundary):
    min_exceed = point < min_boundary
    max_exceed = point > max_boundary
    point = np.max(np.stack([point, min_boundary]), axis=0) # Min boundary
    point = np.min(np.stack([point, max_boundary]), axis=0) # Min boundary
    return point, min_exceed, max_exceed

def move_object(
    obj_coords, max_t:int = 10, focal_length=890, 
    xy_min_boundary=None, xy_max_boundary=None, zmin=None, zmax=None
):
    cur_point = np.array(obj_coords)
    
    vector = np.random.uniform(-1, 1, size=3)
    # accel  = np.array([0., 0., 0.])

    for t in range(max_t):
        min_boundary = np.array([*(xy_min_boundary * cur_point[-1] / focal_length), zmin])
        max_boundary = np.array([*(xy_max_boundary * cur_point[-1] / focal_length), zmax])
        
        dv = 0.1 * np.random.randn(3)
        vector += dv # variate direction
        cur_point = cur_point + vector
        # cur_point = cur_point + vector + accel * t # accleration using 1st derivative
        
        cur_point, min_ex, max_ex = bound_point(cur_point, min_boundary, max_boundary)
        vector[min_ex | max_ex] *= -1 # Invert the direction
        
        yield cur_point

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

def main(args):
    # 1. Load target video and boundaries
    video_lst = []
    boundaries = []
    for input, boundary_path in zip(args.input, args.boundary_path):
        video, fps, width, height = capture_video(input, 100)
        video_lst.append(video)
        
        boundary = load_boundary_from_json(boundary_path) if boundary_path else [0, 0, height, width]
        boundaries.append(boundary)
    height = int(height)
    width = int(width)
    
    # 2. Pixel-system
    world2pixel = width / args.sensor_size[0]
    f_pixel = args.focal_length * world2pixel
    
    xtl, ytl, xbr, ybr = boundaries[args.target_idx]
    # 3. Initialize occluder's camera-system coordinates
    obj, obj_coords = init_object(args, xtl, ytl, xbr, ybr, height, width, f_pixel)
    # 4. Simulate the movement of the occluder
    min_pixel_bnd = np.array([xtl - height/2, ytl - width/2])
    max_pixel_bnd = np.array([xbr - height/2, ybr - width/2])
    movement_generator = move_object(
                                obj_coords, len(video), f_pixel,
                                min_pixel_bnd, max_pixel_bnd, zmin=1e-4, zmax=args.target_distance)
    
    camera_principal_points = [
        np.array(args.camera_principal_point[3*i:3*i+3])
            for i in range(len(args.input))
        ]
    # 5. Project occluder on the original videos
    # for frame_idx, (xt, yt, zt) in enumerate(movement_generator):
    for frame_idx, xyzt in enumerate(movement_generator):
        # Map (xt, yt, zt) to the frontal camera coordinates
        xyzt = np.array(xyzt)
        rot_mat = rotation_matrix(args.camera_rot_type[args.target_idx], args.camera_rot[args.target_idx])
        xyzt = xyzt @ np.linalg.inv(rot_mat) + camera_principal_points[args.target_idx]
        for i, (video, cam_pos, cam_ang) in enumerate(zip(video_lst, args.camera_principal_point, args.camera_rot)):
            # Shift and rotate (xt, yt, zt) for each view
            xyzt_ = xyzt - camera_principal_points[i]
            rot_mat = rotation_matrix(args.camera_rot_type[i], args.camera_rot[i])
            xyzt_ = xyzt_ @ rot_mat
            xt_, yt_, zt_ = xyzt_
            if zt_ < 0:
                continue
            # Project (xt, yt, zt) to the 2D space
            x_img, y_img, _ = xyzt_ * f_pixel / zt_
            x_img += height//2
            y_img += width//2
            # Overlap
            prj_scale = min(f_pixel / zt_, height / obj.shape[0])
            obj_resized = cv2.resize(obj, dsize=(0,0), fx=prj_scale, fy=prj_scale)
            project_occluder_on_image(obj_resized, [x_img, y_img], video[frame_idx])
        
    print("Start saving the output videos..")
    # 7. Save occluded full video
    for output_path, video in zip(args.output, video_lst):
        write_video(output_path, video, fps, width, height)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', nargs='+', help='')
    parser.add_argument('-o', '--output', nargs='+', help='')
    parser.add_argument('-ot', '--object_type', default='box', help='')
    parser.add_argument('-op', '--object_path', required=False, help='')
    parser.add_argument('-os', '--object_size', type=int, help='Occluder maximum size (mm)')
    parser.add_argument('-bp', '--boundary_path', nargs='+', required=False, help='Boundary json filepath')
    
    parser.add_argument('-f', '--focal_length', default=2.86, help='Physical focal length of the target camera (mm)')
    parser.add_argument('-ss', '--sensor_size', nargs=2, default=[6.17, 3.47], help='Physical sensor size (mm)')
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