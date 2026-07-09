from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path


APP_DIR = Path(__file__).resolve().parent


def run_command(command: list[str]) -> None:
    """サブプロセスを実行し、失敗したらその場で停止する。"""
    print(" ".join(command), flush=True)
    subprocess.run(command, cwd=APP_DIR, check=True)


def get_video_size(video_path: Path) -> tuple[int, int]:
    """動画の幅と高さを取得する。"""
    import ffmpeg

    probe = ffmpeg.probe(str(video_path))
    video_stream = next(s for s in probe["streams"] if s["codec_type"] == "video")
    return int(video_stream["width"]), int(video_stream["height"])


def ceil_to_multiple(value: int, multiple: int) -> int:
    """value以上の最小のmultiple倍数を返す。"""
    return int(math.ceil(value / multiple) * multiple)


def prepare_padded_video(video_path: Path, work_dir: Path) -> tuple[Path, Path | None]:
    """必要なら64倍数サイズのProRes LT一時動画を作成する。"""
    original_width, original_height = get_video_size(video_path)
    padded_width = ceil_to_multiple(original_width, 64)
    padded_height = ceil_to_multiple(original_height, 64)

    if original_width == padded_width and original_height == padded_height:
        return video_path, None

    padded_video_path = work_dir / f"{video_path.stem}_padded.mov"
    padding_json_path = work_dir / "padding.json"

    padding = {
        "original_width": original_width,
        "original_height": original_height,
        "padded_width": padded_width,
        "padded_height": padded_height,
        "pad_left": 0,
        "pad_top": 0,
        "pad_right": padded_width - original_width,
        "pad_bottom": padded_height - original_height,
        "padded_video_path": str(padded_video_path),
    }

    if not padded_video_path.exists():
        import ffmpeg

        print(f"padding video: {original_width}x{original_height} -> {padded_width}x{padded_height}", flush=True)
        (
            ffmpeg
            .input(str(video_path))
            .filter("pad", padded_width, padded_height, 0, 0, color="black")
            .output(str(padded_video_path), vcodec="prores_ks", pix_fmt="yuv422p10le", **{"profile:v": "1"})
            .overwrite_output()
            .global_args("-loglevel", "error")
            .run()
        )

    with open(padding_json_path, "w", encoding="utf-8") as f:
        json.dump(padding, f, ensure_ascii=False, indent=2)

    return padded_video_path, padding_json_path


def parse_args() -> argparse.Namespace:
    """一括実行CLIの引数を解析する。"""
    parser = argparse.ArgumentParser(description="Run M2SVID depth, warping, and inpaint pipeline.")
    parser.add_argument("--model_id", default="video-depth-anything-base")
    parser.add_argument("--video_path", required=True)
    parser.add_argument("--disparity_perc", type=float, default=0.1)
    parser.add_argument("--mask_antialias", type=int, default=0)
    parser.add_argument("--model_config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--use_prores", action="store_true", default=False)
    parser.add_argument("--save_sbs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--chunk_size", type=int, default=10)
    parser.add_argument("--depth_input_size", type=int, default=518)
    parser.add_argument("--output_folder", required=True)
    return parser.parse_args()


def main() -> None:
    """入力動画から最終generated動画まで一括実行する。"""
    args = parse_args()

    video_path = Path(args.video_path).resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"動画ファイルが見つかりません: {video_path}")

    output_folder = Path(args.output_folder).resolve()
    work_dir = output_folder / video_path.stem
    model_work_dir = work_dir / args.model_id
    work_dir.mkdir(parents=True, exist_ok=True)
    model_work_dir.mkdir(parents=True, exist_ok=True)
    output_folder.mkdir(parents=True, exist_ok=True)

    processing_video_path, padding_json_path = prepare_padded_video(video_path, work_dir)

    depth_path = model_work_dir / f"{processing_video_path.stem}.npz"
    reprojected_path = model_work_dir / f"{processing_video_path.stem}_reprojected.mp4"
    reprojected_mask_path = model_work_dir / f"{processing_video_path.stem}_reprojected_mask.mp4"

    if not depth_path.exists():
        run_command(
            [
                sys.executable,
                "get_depth.py",
                "--model_id",
                args.model_id,
                "--video_path",
                str(processing_video_path),
                "--save_folder",
                str(model_work_dir),
                "--save_npz",
                "True",
                "--input_size",
                str(args.depth_input_size),
            ]
        )
    else:
        print(f"npz exists. skip depth: {depth_path}", flush=True)

    if not reprojected_path.exists() or not reprojected_mask_path.exists():
        run_command(
            [
                sys.executable,
                "warping.py",
                "--video_path",
                str(processing_video_path),
                "--depth_path",
                str(depth_path),
                "--output_path_reprojected",
                str(reprojected_path),
                "--output_path_mask",
                str(reprojected_mask_path),
                "--disparity_perc",
                str(args.disparity_perc),
            ]
        )
    else:
        print(f"reprojected exists. skip warping: {reprojected_path}", flush=True)

    refine_args = [
        sys.executable,
        "inpaint_and_refine.py",
        "--mask_antialias",
        str(args.mask_antialias),
        "--model_config",
        args.model_config,
        "--ckpt",
        args.ckpt,
        "--video_path",
        str(processing_video_path),
        "--reprojected_path",
        str(reprojected_path),
        "--reprojected_mask_path",
        str(reprojected_mask_path),
        "--output_folder",
        str(output_folder),
        "--output_basename",
        video_path.stem,
        "--chunk_size",
        str(args.chunk_size),
    ]
    if args.use_prores:
        refine_args.append("--use_prores")
    refine_args.append("--save_sbs" if args.save_sbs else "--no-save_sbs")
    if padding_json_path is not None:
        refine_args.extend(["--padding_json_path", str(padding_json_path)])

    run_command(refine_args)


if __name__ == "__main__":
    os.chdir(APP_DIR)
    main()
