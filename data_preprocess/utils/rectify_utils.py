"""
Copyright 2026 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
import torch
import numpy as np
import cv2
import itertools
import sys


def combine_videos_with_lines(left_videos, right_videos, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    left_videos = left_videos.permute(0, 2, 3, 1).cpu().numpy()  # (N, H, W, C)
    right_videos = right_videos.permute(0, 2, 3, 1).cpu().numpy()  # (N, H, W, C)
    
    num_frames, height, width, channels = left_videos.shape

    colors = [
        (255, 0, 0),  
        (0, 255, 0),   
        (0, 0, 255),    
        (255, 255, 0),  
        (255, 0, 255), 
        (0, 255, 255)  
    ]
    
    for i in range(num_frames):
        combined_frame = np.concatenate((left_videos[i], right_videos[i]), axis=1) 

        for idx, y in enumerate(range(0, height, 10)):
            color = colors[idx % len(colors)] 
            cv2.line(combined_frame, (0, y), (combined_frame.shape[1], y), color, 1) 
        
        combined_frame_bgr = cv2.cvtColor(combined_frame, cv2.COLOR_RGB2BGR)

        frame_path = os.path.join(output_folder, f"frame_{i:04d}.png")
        cv2.imwrite(frame_path, combined_frame_bgr)


def compute_rectification_params_loftr(matcher_indoor, matcher_outdoor, video1, video2, output_folder=None, draw_matches=False, 
                            subsample_percent=None, top_k=None, ransac=False):
    if 'compute_feature_matching_loftr' not in sys.modules:
        from utils.loftr import compute_feature_matching_loftr
    pts1, pts2, F, _, _, _, _ = compute_feature_matching_loftr(matcher_indoor, video1, video2, debug_output=output_folder, 
                        draw_matches=draw_matches, subsample_percent=subsample_percent, top_k=top_k, ransac=ransac)
    
    if matcher_outdoor is not None:
        pts1_o, pts2_o, F_o, _, _, _, _ = compute_feature_matching_loftr(matcher_outdoor, video1, video2, debug_output=output_folder, 
                            draw_matches=draw_matches, subsample_percent=subsample_percent, top_k=top_k, ransac=ransac)

        if len(pts1_o) > len(pts1):
            print(f'outdoor has more matched points {len(pts1_o)} > {len(pts1)}, using outdoor')
            pts1, pts2, F = pts1_o, pts2_o, F_o
        else:
            print(f'indoor has more matched points {len(pts1)} > {len(pts1_o)}, using indoor')

    _, _, h, w = video1.shape
    retval, H1, H2 = cv2.stereoRectifyUncalibrated(pts1.ravel(), pts2.ravel(), F, [w, h])

    if (retval == False):
        print("ERROR: stereoRectifyUncalibrated failed")
        raise ValueError

    valid_region = compute_valid_region(H1, H2, [h, w], [h, w])

    return H1, H2, valid_region


def rectify_videos(video1, video2, rectification_params, top_k=5, output_folder=None, ransacReprojThreshold=1.0):
    H1, H2, valid_region = rectification_params

    if output_folder is not None:
        os.makedirs(output_folder, exist_ok=True)

    rectified_frames1 = []
    rectified_frames2 = []

    for i, (frame1, frame2) in enumerate(zip(video1, video2)):
        rectified_frame1 = cv2.warpPerspective(frame1, H1, (frame1.shape[1], frame1.shape[0]))
        rectified_frame2 = cv2.warpPerspective(frame2, H2, (frame2.shape[1], frame2.shape[0]))

        if output_folder is not None:
            cv2.imwrite(os.path.join(output_folder, f"rectified_full_left_{i:04d}.png"), rectified_frame1)
            cv2.imwrite(os.path.join(output_folder, f"rectified_full_right_{i:04d}.png"), rectified_frame2)
            
        x_min, y_min, x_max, y_max = valid_region
        rectified_frame1 = rectified_frame1[y_min:y_max, x_min:x_max]
        rectified_frame2 = rectified_frame2[y_min:y_max, x_min:x_max]

        rectified_frames1.append(rectified_frame1)
        rectified_frames2.append(rectified_frame2)

    if output_folder is not None:
        for i, (frame1, frame2) in enumerate(zip(rectified_frames1, rectified_frames2)):
            cv2.imwrite(os.path.join(output_folder, f"rectified_left_{i:04d}.png"), frame1)
            cv2.imwrite(os.path.join(output_folder, f"rectified_right_{i:04d}.png"), frame2)

    return rectification_params, rectified_frames1, rectified_frames2


def compute_valid_region(H1, H2, img_shape1, img_shape2):
    def compute_valid_region(H, img_shape):
        height, width = img_shape

        # Define the four corners of the original image
        corners = np.array([
            [0, 0],
            [width, 0],
            [width, height],
            [0, height]
        ], dtype=np.float32).reshape(-1, 1, 2)  # Shape (4, 1, 2) for perspectiveTransform

        # Transform the corners using the homography
        transformed_corners = cv2.perspectiveTransform(corners, H).reshape(-1, 2)  # Shape (4, 2)
        x_coords, y_coords = transformed_corners[:, 0], transformed_corners[:, 1]

        y_min = np.max(y_coords[:2])
        y_max = np.min(y_coords[2:])
        
        x_min = max(x_coords[0], x_coords[3])
        x_max = min(x_coords[1], x_coords[2])

        # Clip to ensure bounding box is within image bounds
        x_min = max(x_min, 0)
        y_min = max(y_min, 0)
        x_max = min(x_max, width)
        y_max = min(y_max, height)

        return int(np.ceil(x_min)), int(np.ceil(y_min)), int(np.floor(x_max)), int(np.floor(y_max))
  
    x_min1, y_min1, x_max1, y_max1 = compute_valid_region(H1, img_shape1)
    x_min2, y_min2, x_max2, y_max2 = compute_valid_region(H2, img_shape2)
    x_min = max(x_min1, x_min2)
    y_min = max(y_min1, y_min2)
    x_max = min(x_max1, x_max2)
    y_max = min(y_max1, y_max2)
    # print(x_min, y_min, x_max, y_max)
    # nessesary in order that ffmpeg can save video
    y_max = int((y_max - y_min) // 2 * 2) + y_min
    x_max = int((x_max - x_min) // 2 * 2) + x_min
    
    return x_min, y_min, x_max, y_max 


def torch_to_opencv_format(video_tensor):
    video_np = video_tensor.permute(0, 2, 3, 1).cpu().numpy()
    if video_np.dtype != np.uint8:
        video_np = (video_np * 255).astype(np.uint8)
    video_np = [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in video_np]
    return video_np
    

def opencv_to_torch_format(frames):
    frames_rgb = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
    video_np = np.stack(frames_rgb, axis=0)
    video_tensor = torch.from_numpy(video_np).permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
    
    return video_tensor


def opencv_to_numpy_format(frames):
    frames_rgb = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
    video_np = np.stack(frames_rgb, axis=0)
    return video_np