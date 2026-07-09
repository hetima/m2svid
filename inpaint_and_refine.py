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
import gc
import json

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

# PYTHONPATH="./:./third_party/Hi3D_Official/:./third_party/pytorch_msssim/:${PYTHONPATH}"
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
for _p in [
    _SCRIPT_DIR,
    os.path.join(_SCRIPT_DIR, "third_party", "Hi3D_Official"),
    os.path.join(_SCRIPT_DIR, "third_party", "pytorch_msssim"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import random
import argparse
from datetime import datetime
from pytorch_lightning import seed_everything
import ffmpeg
from torchvision import transforms
import torch
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf
from third_party.Hi3D_Official.sgm.util import instantiate_from_config

from m2svid.utils.video_utils import open_ffmpeg_process, get_video_fps
from m2svid.data.utils import get_video_frames, apply_closing, apply_dilation
from m2svid.utils.anaglyph import make_anaglyph_video


def log_memory(label: str) -> None:
    """現在のCPU/GPUメモリ使用量をログへ出す。"""
    parts = [f"[memory] {label}"]
    try:
        import psutil
        rss_gb = psutil.Process(os.getpid()).memory_info().rss / 1024**3
        parts.append(f"rss={rss_gb:.2f}GiB")
    except Exception:
        pass
    if torch.cuda.is_available():
        allocated_gb = torch.cuda.memory_allocated() / 1024**3
        reserved_gb = torch.cuda.memory_reserved() / 1024**3
        parts.append(f"cuda_allocated={allocated_gb:.2f}GiB")
        parts.append(f"cuda_reserved={reserved_gb:.2f}GiB")
    print(" ".join(parts), flush=True)

parser = argparse.ArgumentParser()
parser.add_argument("--model_config", type=str)
parser.add_argument("--ckpt", type=str)
parser.add_argument("--video_path", type=str)
parser.add_argument("--reprojected_path", type=str)
parser.add_argument("--reprojected_mask_path", type=str)
parser.add_argument("--output_folder", type=str)
parser.add_argument("--output_basename", type=str, default=None, help="Output filename basename (default: derived from video_path)")
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
parser.add_argument("--chunk_size", type=int, default=10, help="Number of target frames to generate per chunk (VRAM reduction)")
parser.add_argument("--use_prores", action="store_true", default=False, help="Save videos as ProRes 422 LT .mov")
parser.add_argument("--padding_json_path", type=str, default=None, help="Path to padding metadata JSON")
# --overlap は効果が薄い上にやたら時間がかかるので0推奨
parser.add_argument("--overlap", type=int, default=0, help="Number of overlapping frames on each side of a chunk for temporal continuity")
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
    log_memory("before model instantiate")
    denoising_model = instantiate_from_config(config.model).half()
    log_memory("after model instantiate")
    quantize(denoising_model, weights=qint8, exclude=["first_stage_model*"])
    freeze(denoising_model)
    log_memory("after quanto freeze")
    denoising_model.init_from_ckpt(args.ckpt)
    gc.collect()
    log_memory("after checkpoint load")
    denoising_model = denoising_model.cuda().eval()
    log_memory("after cuda")
else:
    log_memory("before model instantiate")
    denoising_model = instantiate_from_config(config.model).half()
    log_memory("after model instantiate")
    denoising_model.init_from_ckpt(args.ckpt, assign=True)
    denoising_model = denoising_model.half()
    gc.collect()
    log_memory("after checkpoint load")
    denoising_model = denoising_model.cuda().eval()
    log_memory("after cuda")

reprojected_closing_holes_kernel = args.reprojected_closing_holes_kernel
mask_antialias = args.mask_antialias
output_folder = args.output_folder

# load and preprocess videos (probe once, reuse)
log_memory("before video load")
video_probe = ffmpeg.probe(args.video_path)
fps = get_video_fps(args.video_path, video_probe)

input_video = get_video_frames(args.video_path)
reprojected = get_video_frames(args.reprojected_path)
reprojected_mask = get_video_frames(args.reprojected_mask_path, video_is_grayscale=True)
log_memory("after video load")

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
log_memory("after video preprocess")


chunk_size = args.chunk_size
overlap = args.overlap
assert chunk_size + 2 * overlap <= denoising_model.num_samples, (
    f"chunk_size({chunk_size}) + 2*overlap({overlap}) = {chunk_size + 2 * overlap} "
    f"must be <= num_samples({denoising_model.num_samples})"
)
stride = chunk_size  # target frames to advance per chunk
num_chunks = max(1, (t + stride - 1) // stride)  # ceil(t / stride)

generated_chunks = []

with torch.inference_mode():
    pbar = tqdm(range(num_chunks), desc="Generating chunks")
    for chunk_idx in pbar:
        # --- compute target frame range (what we want to keep) ---
        tgt_start = chunk_idx * stride
        tgt_end = min(tgt_start + chunk_size, t)

        # --- compute input range including overlap ---
        inp_start = max(0, tgt_start - overlap)
        inp_end = min(tgt_end + overlap, t)
        actual_overlap_left = tgt_start - inp_start
        actual_overlap_right = inp_end - tgt_end

        pbar.set_description(
            # f"chunk {chunk_idx + 1}/{num_chunks}, "
            f"[input {inp_start}-{inp_end - 1}, "
            f"output {tgt_start}-{tgt_end - 1}]"
        )

        chunk_batch = {
            "video": input_video[None, :, inp_start:inp_end].cuda(),
            "video_2nd_view": input_video[None, :, inp_start:inp_end].cuda(),
            "reprojected_video": reprojected[None, :, inp_start:inp_end].cuda(),
            "reprojected_mask": reprojected_mask[None, :, inp_start:inp_end].cuda(),
            "fps_id": torch.tensor([fps]).cuda(),
            "caption": [""],
            "motion_bucket_id": torch.tensor([127]).cuda(),
        }

        chunk_output = denoising_model.generate(chunk_batch)["generated-video"]
        # chunk_output shape: [1, C, T_input, H, W]
        # Extract only the target (center) frames
        center_start = actual_overlap_left
        center_end = chunk_output.shape[2] - actual_overlap_right
        generated_chunks.append(chunk_output[0, :, center_start:center_end].cpu())
        del chunk_batch, chunk_output

generated_video = torch.cat(generated_chunks, dim=1)  # [c, t_total, h, w]
del generated_chunks, reprojected, reprojected_mask, denoising_model
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()


def check_unique_path(path: str) -> str:
    """pathのファイルが存在する場合、_01, _02, ... のサフィックスを付けてユニークなパスを返す。"""
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    for i in range(1, 99):
        candidate = f"{base}_{i:02}{ext}"
        if not os.path.exists(candidate):
            return candidate
    return path


def check_unique_paths(files: list) -> str:
    """filesのリストをチェックし、必要な最大のサフィックス文字列を返す。

    いずれのファイルも存在しなければ空文字列を返す。
    ファイルが存在する場合、リスト内で最も大きい _NN サフィックスを返す。
    99を超える場合は _YYYYMMDD_HHMMSS 形式のタイムスタンプを返す。
    """
    max_suffix = 0
    any_exists = False
    for path in files:
        if not os.path.exists(path):
            continue
        any_exists = True
        base, ext = os.path.splitext(path)
        for i in range(1, 100):
            candidate = f"{base}_{i:02}{ext}"
            if not os.path.exists(candidate):
                if i - 1 > max_suffix:
                    max_suffix = i - 1
                break
        else:
            # 99まで全て存在
            max_suffix = 99

    if not any_exists:
        return ""

    if max_suffix >= 99:
        return datetime.now().strftime("_%Y%m%d_%H%M%S")

    return f"_{max_suffix + 1:02d}"


def frame_to_uint8(frame):
    """[-1, 1] の1フレームテンソルを ffmpeg 用の uint8 RGB 配列に変換する。"""
    return (
        ((frame.detach().cpu().float() + 1) / 2)
        .clamp(0, 1)
        .mul(255)
        .byte()
        .permute(1, 2, 0)
        .numpy()
    )


def save_video(video, fps, path):
    """動画全体をNumPy化せず、1フレームずつffmpegへ書き出す。"""
    video = video[0]  # [1, C, T, H, W] -> [C, T, H, W]
    _, t, h, w = video.shape
    use_prores = os.path.splitext(path)[1].lower() == ".mov"
    process = open_ffmpeg_process(path, w, h, fps, crf=17, use_prores=use_prores)
    for i in range(t):
        process.stdin.write(frame_to_uint8(video[:, i]).tobytes())
    process.stdin.close()
    process.wait()


def save_sbs_video(left_video, right_video, fps, path):
    """SBS動画を全体結合せず、左右1フレームずつ横結合して書き出す。"""
    _, t, h, w = left_video.shape
    use_prores = os.path.splitext(path)[1].lower() == ".mov"
    process = open_ffmpeg_process(path, w * 2, h, fps, crf=17, use_prores=use_prores)
    for i in range(t):
        left_frame = frame_to_uint8(left_video[:, i])
        right_frame = frame_to_uint8(right_video[:, i])
        process.stdin.write(np.concatenate([left_frame, right_frame], axis=1).tobytes())
    process.stdin.close()
    process.wait()


def crop_padding(video, padding_json_path):
    """padding.json が指定されていれば、動画テンソルを元の解像度へ切り戻す。"""
    if not padding_json_path:
        return video
    with open(padding_json_path, "r", encoding="utf-8") as f:
        padding = json.load(f)

    top = int(padding.get("pad_top", 0))
    left = int(padding.get("pad_left", 0))
    original_height = int(padding["original_height"])
    original_width = int(padding["original_width"])
    return video[:, :, top : top + original_height, left : left + original_width]


video_name = args.output_basename if args.output_basename else os.path.splitext(os.path.basename(args.video_path))[0]
os.makedirs(output_folder, exist_ok=True)

# 3種の出力パスを定義
output_ext = ".mov" if args.use_prores else ".mp4"
gen_path = os.path.join(output_folder, f"{video_name}_generated{output_ext}")
sbs_path = os.path.join(output_folder, f"{video_name}_sbs{output_ext}")
anaglyph_path = os.path.join(output_folder, f"{video_name}_anaglyph{output_ext}")

# 統一サフィックスを決定
paths_to_check = [gen_path, sbs_path, anaglyph_path]
suffix = check_unique_paths(paths_to_check)

if suffix:
    base_gen, ext_gen = os.path.splitext(gen_path)
    gen_path = f"{base_gen}{suffix}{ext_gen}"
    base_sbs, ext_sbs = os.path.splitext(sbs_path)
    sbs_path = f"{base_sbs}{suffix}{ext_sbs}"
    base_ana, ext_ana = os.path.splitext(anaglyph_path)
    anaglyph_path = f"{base_ana}{suffix}{ext_ana}"

input_video = crop_padding(input_video, args.padding_json_path)
generated_video = crop_padding(generated_video, args.padding_json_path)

save_video(generated_video[None], fps, gen_path)
gc.collect()

if args.save_sbs:
    save_sbs_video(input_video, generated_video, fps, sbs_path)
    gc.collect()

if args.save_anaglyph:
    anaglyph = make_anaglyph_video(
        input_video, generated_video, unnormalized_videos=True
    )
    save_video(anaglyph[None], fps, anaglyph_path)
