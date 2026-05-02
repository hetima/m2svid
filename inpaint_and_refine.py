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

# PYTHONPATH="./:./third_party/Hi3D-Official/:./third_party/pytorch-msssim/:${PYTHONPATH}"
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
for _p in [
    _SCRIPT_DIR,
    os.path.join(_SCRIPT_DIR, "third_party", "Hi3D-Official"),
    os.path.join(_SCRIPT_DIR, "third_party", "pytorch-msssim"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import random
import argparse
from pytorch_lightning import seed_everything
import ffmpeg
from torchvision import transforms
import torch
import numpy as np
import torchvision.io
from omegaconf import OmegaConf
from sgm.util import instantiate_from_config

from m2svid.utils.video_utils import open_ffmpeg_process, get_video_fps
from m2svid.data.utils import get_video_frames, apply_closing, apply_dilation
from m2svid.utils.anaglyph import make_anaglyph_video

parser = argparse.ArgumentParser()
parser.add_argument("--model_config", type=str)
parser.add_argument("--ckpt", type=str)
parser.add_argument("--video_path", type=str)
parser.add_argument("--reprojected_path", type=str)
parser.add_argument("--reprojected_mask_path", type=str)
parser.add_argument("--output_folder", type=str)
parser.add_argument("--reprojected_closing_holes_kernel", type=int, default=11)
parser.add_argument("--mask_antialias", type=int, default=False)
parser.add_argument(
    "--save_sbs",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Save side-by-side video",
)
parser.add_argument("--save_anaglyph", action="store_true", help="Save anaglyph video")
parser.add_argument("--enable_vae_fp16", action="store_true", default=False, help="Enable FP16 autocast for VAE (may reduce quality)")
parser.add_argument("--quanto_int8", action="store_true", default=False, help="Load optimum-quanto int8 quantized checkpoint (reduces GPU memory)")
args = parser.parse_args()

# ckptファイル名に"quanto_int8"が含まれていれば自動的に有効化
if "quanto_int8" in os.path.basename(args.ckpt):
    args.quanto_int8 = True


seed = random.randint(0, 65535)
seed_everything(seed)

config = OmegaConf.load(args.model_config)
# Override VAE autocast setting based on CLI flag (default: VAE runs in FP32 for stability)
config.model.params.disable_first_stage_autocast = not args.enable_vae_fp16

if args.quanto_int8:
    # optimum-quanto int8量子化済みモデルの読み込み
    # first_stage_model (VAE) は除外: Conv2dがtimesteps kwargを使うためQConv2dと非互換
    from optimum.quanto import quantize, freeze, qint8
    denoising_model = instantiate_from_config(config.model).half()
    quantize(denoising_model, weights=qint8, exclude=["first_stage_model*"])
    freeze(denoising_model)
    denoising_model.init_from_ckpt(args.ckpt)
    denoising_model = denoising_model.cuda().eval()
else:
    denoising_model = instantiate_from_config(config.model).half()
    denoising_model.init_from_ckpt(args.ckpt)
    denoising_model = denoising_model.cuda().eval()

reprojected_closing_holes_kernel = args.reprojected_closing_holes_kernel
mask_antialias = args.mask_antialias
output_folder = args.output_folder

# load and preprocess videos (probe once, reuse)
video_probe = ffmpeg.probe(args.video_path)
fps = get_video_fps(args.video_path, video_probe)

input_video = get_video_frames(args.video_path)
reprojected = get_video_frames(args.reprojected_path)
reprojected_mask = get_video_frames(args.reprojected_mask_path, video_is_grayscale=True)

reprojected_mask = apply_closing(reprojected_mask, reprojected_closing_holes_kernel)
reprojected[reprojected_mask.repeat(1, 3, 1, 1) > 0.5] = 0
reprojected_mask = apply_dilation(reprojected_mask, 3)
reprojected_mask = reprojected_mask.repeat(1, 3, 1, 1)

input_video = input_video.permute(1, 0, 2, 3).float() * 2 - 1  # [t,c,h,w] -> [c,t,h,w]
reprojected = reprojected.permute(1, 0, 2, 3).float() * 2 - 1  # [t,c,h,w] -> [c,t,h,w]
reprojected_mask = (
    reprojected_mask.permute(1, 0, 2, 3).float() * 2 - 1
)  # [t,c,h,w] -> [c,t,h,w]

c, t, h, w = reprojected_mask.shape
downsampled_resolution = [int(h / 8), int(w / 8)]
reprojected_mask = reprojected_mask.permute(
    1, 0, 2, 3
).float()  # [c,t,h,w] -> [t,c,h,w]
reprojected_mask = transforms.Resize(downsampled_resolution, antialias=mask_antialias)(
    reprojected_mask
)
reprojected_mask = reprojected_mask[:, [0]]
reprojected_mask = reprojected_mask.permute(
    1, 0, 2, 3
).float()  # [t,c,h,w] -> [c,t,h,w]


chunk_size = denoising_model.num_samples  # 25
num_chunks = (t + chunk_size - 1) // chunk_size

generated_chunks = []

with torch.inference_mode():
    for chunk_idx in range(num_chunks):
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, t)
        print(
            f"Processing chunk {chunk_idx + 1}/{num_chunks}, frames {start}-{end - 1}"
        )

        chunk_batch = {
            "video": input_video[None, :, start:end].cuda(),
            "video_2nd_view": input_video[None, :, start:end].cuda(),
            "reprojected_video": reprojected[None, :, start:end].cuda(),
            "reprojected_mask": reprojected_mask[None, :, start:end].cuda(),
            "fps_id": torch.tensor([fps]).cuda(),
            "caption": [""],
            "motion_bucket_id": torch.tensor([127]).cuda(),
        }

        chunk_output = denoising_model.generate(chunk_batch)["generated-video"]
        generated_chunks.append(chunk_output[0].cpu())

generated_video = torch.cat(generated_chunks, dim=1)  # [c, t_total, h, w]


def save_video(video, fps, path):
    frames = video.cpu().numpy().transpose(0, 2, 3, 4, 1)
    frames = np.concatenate(frames)
    frames = (((frames + 1) / 2).clip(0, 1) * 255).astype(np.uint8)
    torchvision.io.write_video(path, frames, fps=int(fps), options={"crf": "17"})


video_name = os.path.splitext(os.path.basename(args.video_path))[0]
os.makedirs(output_folder, exist_ok=True)
save_video(
    generated_video[None],
    fps,
    os.path.join(output_folder, f"{video_name}_generated.mp4"),
)

if args.save_sbs:
    sbs_video = torch.cat([input_video, generated_video], dim=-1)
    save_video(
        sbs_video[None], fps, os.path.join(output_folder, f"{video_name}_sbs.mp4")
    )

if args.save_anaglyph:
    anaglyph = make_anaglyph_video(
        input_video, generated_video, unnormalized_videos=True
    )
    save_video(
        anaglyph[None], fps, os.path.join(output_folder, f"{video_name}_anaglyph.mp4")
    )
