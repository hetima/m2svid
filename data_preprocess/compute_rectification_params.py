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
import traceback

import json
import ffmpeg
import tqdm

from m2svid.utils.video_utils import get_video_fps, get_total_frames, get_video_frames, split_left_right
from utils.rectify_utils import compute_rectification_params_loftr


def preprocess_rectification_params(name, REWRITE):
    try:
        video_path = f'{video_root}/{name}'
        if debug_root is not None:
            debug_output_path = f'{debug_root}/{name}'
        else:
            debug_output_path = None

        output_path = f'{output_root}/{name}.json'
        if os.path.exists(output_path):
            print(f'Video: {name} is already processed.')
            if REWRITE:
                print(f'Rewriting video: {name} .')
                os.remove(output_path)
            else:
                return
        else:
            print(f'Processing {name}.')

        probe = ffmpeg.probe(video_path)
        fps = get_video_fps(video_path, probe)
        total_frames = get_total_frames(video_path, probe)
        duration = total_frames / fps
        shift_fps = max(1, num_frames / duration)

        frames = get_video_frames(video_path, fps=shift_fps, num_frames=num_frames)
        frames = frames.transpose(0, 3, 1, 2)
        frames = torch.from_numpy(frames)

        left_videos, right_videos = split_left_right(frames, rectified=False)

        rectification_params = compute_rectification_params_loftr(matcher_indoor, matcher_outdoor, left_videos, right_videos, draw_matches=True, output_folder=debug_output_path,
                                                                top_k=top_k,
                                                                ransac=ransac
                                                                )
        rectification_params = {
            'homography_left': rectification_params[0].tolist(),
            'homography_right': rectification_params[1].tolist(),
            'croping': rectification_params[2],

        }

        with open(output_path, 'w') as fout:
            json.dump(rectification_params, fout)
    except Exception:
        print(f'--------------- Error processing video: {name} --------------------')
        traceback.print_exc()



with open('datasets/ego4d/subsets/all_full_videos.json') as fin:
    data = json.load(fin)
dataset_root = 'datasets/ego4d'
video_root = f'{dataset_root}/videos/'
label_dict = None

REWRITE = False
debug_root = None
ransac = True

import kornia.feature as KF

output_root = f'{dataset_root}/recrification_params_loftr_ransac'
# debug_root = f'{dataset_root}/examples/compute_rectification_params//'
matcher_outdoor = KF.LoFTR(pretrained="outdoor").cuda()
matcher_indoor = KF.LoFTR(pretrained="indoor_new").cuda()

top_k = None
num_frames = 200

os.makedirs(output_root, exist_ok=True)
np.random.shuffle(data)

for name in tqdm.tqdm(data):
    preprocess_rectification_params(name, REWRITE)


