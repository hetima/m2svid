# Copyright 2026 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


#!/bin/bash

set -x

source /opt/conda/bin/activate ""
conda activate depthcrafter
PYTHONPATH="third_party/DepthCrafter/::${PYTHONPATH}" python third_party/DepthCrafter/run.py  \
        --video-path demo/input.mp4 --save_folder outputs/depthcrafter --save_npz True --num_inference_steps 25 --max_res 1024 


source /opt/conda/bin/activate ""
conda activate sgm
PYTHONPATH="./:./third_party/Hi3D_Official/:./third_party/pytorch_msssim/:${PYTHONPATH}" python warping.py  \
        --video_path demo/input.mp4 \
        --depth_path outputs/depthcrafter/input.npz \
        --output_path_reprojected outputs/reprojected/input_reprojected.mp4  \
        --output_path_mask outputs/reprojected/input_reprojected_mask.mp4 \
        --disparity_perc 0.05


source /opt/conda/bin/activate ""
conda activate sgm
PYTHONPATH="./:./third_party/Hi3D_Official/:./third_party/pytorch_msssim/:${PYTHONPATH}" python inpaint_and_refine.py  \
        --mask_antialias 0 \
        --model_config configs/m2svid.yaml \
        --ckpt ckpts/m2svid_weights.pt \
        --video_path demo/input.mp4  \
        --reprojected_path outputs/reprojected/input_reprojected.mp4 \
        --reprojected_mask_path outputs/reprojected/input_reprojected_mask.mp4\
        --output_folder outputs/m2svid \
        