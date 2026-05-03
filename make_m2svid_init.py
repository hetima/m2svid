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

import sys
import os

# PYTHONPATH="./:./third_party/Hi3D_Official/:./third_party/pytorch_msssim/:${PYTHONPATH}"
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
for _p in [
    _SCRIPT_DIR,
    os.path.join(_SCRIPT_DIR, "third_party", "Hi3D_Official"),
    os.path.join(_SCRIPT_DIR, "third_party", "pytorch_msssim"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
from safetensors.torch import load_file as load_safetensors
from safetensors.torch import load_model, save_model

from omegaconf import OmegaConf
from third_party.Hi3D_Official.sgm.util import instantiate_from_config

config_path = 'configs/m2svid.yaml'
input_path = 'ckpts/stable-video-diffusion-img2vid-xt/svd_xt.safetensors'
output_path = 'ckpts/stable-video-diffusion-img2vid-xt/svd_xt_vid2vid_13ch.safetensors'
output_path = 'ckpts/stable-video-diffusion-img2vid-xt/svd_xt_vid2vid_13ch_tmp2.safetensors'

assert not os.path.exists(output_path), f'Output filename already exists: {output_path}'
assert os.path.exists(os.path.dirname(output_path)), f'Output path is not valid: {output_path}'

config = OmegaConf.load(config_path)
model = instantiate_from_config(config.model).cpu()
svd_ckpt = load_safetensors(input_path)
scratch_dict = model.state_dict()

target_dict = {}
for k in scratch_dict.keys():
    if k in svd_ckpt:
        weights = svd_ckpt[k].clone()
        if 'diffusion_model.input_blocks.0.0.weight' in k:
            weights_ex = [weights[:, :4]]
            N = 2
            for _ in range(N):
                weights_ex.append(weights[:, 4:8] / 2)
            weights_ex.append(torch.zeros_like(weights[:, :1]))
            weights = torch.cat(weights_ex, 1)
            print("New shape:", weights.shape)
    elif 'conditioner.embedders.5' in k:
        weights = svd_ckpt[k.replace('conditioner.embedders.5', 'conditioner.embedders.3')].clone()
    else:
        print(f'New weights: {k}')
        weights = scratch_dict[k].clone()
    target_dict[k] = weights
model.load_state_dict(target_dict, strict=True)

if 'safetensors' in output_path:
    save_model(model, output_path)
else:
    torch.save(model.state_dict(), output_path)
