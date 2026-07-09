from __future__ import annotations

import argparse
import gc
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch


APP_DIR = Path(__file__).resolve().parent
VDA_DIR = APP_DIR / "third_party" / "Video_Depth_Anything"

VDA_MODEL_CONFIGS = {
    "video-depth-anything-small": {
        "encoder": "vits",
        "features": 64,
        "out_channels": [48, 96, 192, 384],
    },
    "video-depth-anything-base": {
        "encoder": "vitb",
        "features": 128,
        "out_channels": [96, 192, 384, 768],
    },
    "video-depth-anything-large": {
        "encoder": "vitl",
        "features": 256,
        "out_channels": [256, 512, 1024, 1024],
    },
}


def str_to_bool(value: str | bool) -> bool:
    """CLIのTrue/False文字列をboolへ変換する。"""
    if isinstance(value, bool):
        return value
    normalized = value.lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"bool値として解釈できません: {value}")


def normalize_depth(depths: np.ndarray) -> np.ndarray:
    """DepthCrafter互換の0-1相対深度へ正規化する。"""
    depths = depths.astype(np.float32, copy=False)
    depth_min = float(depths.min())
    depth_max = float(depths.max())
    if depth_max <= depth_min:
        return np.zeros_like(depths, dtype=np.float32)
    return (depths - depth_min) / (depth_max - depth_min)


def get_vda_input_size_candidates(input_size: int) -> list[int]:
    """OOM時に試すVDA入力サイズ候補を作る。"""
    candidates = [input_size, 448, 392, 336, 280]
    return sorted({size for size in candidates if size <= input_size}, reverse=True)


def run_depthcrafter(args: argparse.Namespace) -> None:
    """既存のDepthCrafter CLIを呼び出す。"""
    command = [
        sys.executable,
        "third_party/DepthCrafter/run.py",
        "--video_path",
        args.video_path,
        "--save_folder",
        args.save_folder,
        "--save_npz",
        str(args.save_npz),
        "--num_inference_steps",
        str(args.num_inference_steps),
        "--max_res",
        str(args.max_res),
    ]
    if args.process_length != -1:
        command.extend(["--process_length", str(args.process_length)])
    if args.target_fps != -1:
        command.extend(["--target_fps", str(args.target_fps)])
    subprocess.run(command, cwd=APP_DIR, check=True)


def run_video_depth_anything(args: argparse.Namespace) -> None:
    """Video-Depth-Anythingで深度を推定し、既存warping互換のnpzを保存する。"""
    if not VDA_DIR.exists():
        raise FileNotFoundError(f"Video-Depth-Anythingが見つかりません: {VDA_DIR}")

    sys.path.insert(0, str(VDA_DIR))
    from video_depth_anything.video_depth import VideoDepthAnything
    from utils.dc_utils import read_video_frames, save_video

    model_config = VDA_MODEL_CONFIGS[args.model_id]
    encoder = model_config["encoder"]
    checkpoint_path = APP_DIR / "ckpts" / f"video_depth_anything_{encoder}.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"VDAチェックポイントが見つかりません: {checkpoint_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VideoDepthAnything(**model_config, metric=False)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"), strict=True)
    model = model.to(device).eval()

    frames, target_fps = read_video_frames(
        args.video_path,
        args.process_length,
        args.target_fps,
        args.max_res,
    )
    input_size_candidates = get_vda_input_size_candidates(args.input_size)
    for i, input_size in enumerate(input_size_candidates):
        try:
            print(f"VDA input_size={input_size} で深度推定します。", flush=True)
            with torch.inference_mode():
                depths, fps = model.infer_video_depth(
                    frames,
                    target_fps,
                    input_size=input_size,
                    device=device,
                    fp32=args.fp32,
                )
            break
        except torch.OutOfMemoryError:
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            if i == len(input_size_candidates) - 1:
                raise
            print(
                f"CUDA OOM のため input_size を下げて再試行します: "
                f"{input_size} -> {input_size_candidates[i + 1]}",
                flush=True,
            )

    depths = normalize_depth(np.asarray(depths))

    save_folder = Path(args.save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)
    save_path = save_folder / Path(args.video_path).stem

    save_video(frames, str(save_path) + "_input.mp4", fps=fps)
    save_video(depths, str(save_path) + "_vis.mp4", fps=fps, is_depths=True)
    save_video(depths, str(save_path) + "_depth.mp4", fps=fps, is_depths=True, grayscale=True)

    if args.save_npz:
        np.savez_compressed(str(save_path) + ".npz", depth=depths)


def parse_args() -> argparse.Namespace:
    """深度推定CLIの引数を解析する。"""
    parser = argparse.ArgumentParser(description="Generate depth npz for M2SVID.")
    parser.add_argument("--model_id", default="depthcrafter", choices=["depthcrafter", *VDA_MODEL_CONFIGS.keys()])
    parser.add_argument("--video_path", required=True)
    parser.add_argument("--save_folder", default="./demo_output")
    parser.add_argument("--save_npz", type=str_to_bool, default=False)
    parser.add_argument("--num_inference_steps", type=int, default=5)
    parser.add_argument("--max_res", type=int, default=1024)
    parser.add_argument("--process_length", type=int, default=-1)
    parser.add_argument("--target_fps", type=int, default=-1)
    parser.add_argument("--input_size", type=int, default=518)
    parser.add_argument("--fp32", action="store_true")
    return parser.parse_args()


def main() -> None:
    """指定モデルで深度を推定する。"""
    args = parse_args()
    if args.model_id == "depthcrafter":
        run_depthcrafter(args)
    else:
        run_video_depth_anything(args)


if __name__ == "__main__":
    os.chdir(APP_DIR)
    main()
