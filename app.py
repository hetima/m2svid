from __future__ import annotations

import shutil
import subprocess
import sys
import time
from pathlib import Path

import gradio as gr


APP_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = APP_DIR / "outputs"
CONFIG_PATH = APP_DIR / "configs" / "m2svid_combined.yaml"
CHECKPOINTS = {
    "fp16": APP_DIR / "ckpts" / "m2svid_combined" / "m2svid_combined_fp16.safetensors",
    "int8": APP_DIR / "ckpts" / "m2svid_combined" / "m2svid_combined_quanto_int8.safetensors",
}


def run_command(args: list[str]) -> None:
    """外部コマンドをリポジトリルート基準で実行する。"""
    subprocess.run(args, cwd=APP_DIR, check=True)


def find_output_video(base_name: str, suffix: str, started_at: float) -> Path | None:
    """今回の実行で生成された動画を探す。"""
    candidates = sorted(
        OUTPUT_ROOT.glob(f"{base_name}_{suffix}*.mp4"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for path in candidates:
        if path.stat().st_mtime >= started_at:
            return path
    return candidates[0] if candidates else None


def convert_video(video_path: str | None, chunk_size: int, checkpoint_kind: str, progress=gr.Progress()):
    """アップロード動画からM2SVIDの生成動画を作成する。"""
    if not video_path:
        raise gr.Error("動画をアップロードしてください。")

    input_path = Path(video_path).resolve()
    if not input_path.exists():
        raise gr.Error("動画ファイルが見つかりません。")

    checkpoint_path = CHECKPOINTS[checkpoint_kind]
    if not checkpoint_path.exists():
        raise gr.Error(f"チェックポイントが見つかりません: {checkpoint_path}")

    started_at = time.time()
    base_name = input_path.stem
    project_dir = OUTPUT_ROOT / base_name
    project_dir.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    npz_path = project_dir / f"{base_name}.npz"
    reprojected_path = project_dir / f"{base_name}_reprojected.mp4"
    reprojected_mask_path = project_dir / f"{base_name}_reprojected_mask.mp4"

    if not npz_path.exists():
        progress(0.05, desc="DepthCrafterを実行中")
        run_command(
            [
                sys.executable,
                "third_party/DepthCrafter/run.py",
                "--video-path",
                str(input_path),
                "--save_folder",
                str(project_dir),
                "--save_npz",
                "True",
                "--num_inference_steps",
                "5",
                "--max_res",
                "1024",
            ]
        )

    if not reprojected_path.exists() or not reprojected_mask_path.exists():
        progress(0.35, desc="Warpingを実行中")
        run_command(
            [
                sys.executable,
                "warping.py",
                "--video_path",
                str(input_path),
                "--depth_path",
                str(npz_path),
                "--output_path_reprojected",
                str(reprojected_path),
                "--output_path_mask",
                str(reprojected_mask_path),
                "--disparity_perc",
                "0.1",
            ]
        )

    progress(0.65, desc="Inpaint and refineを実行中")
    run_command(
        [
            sys.executable,
            "inpaint_and_refine.py",
            "--mask_antialias",
            "0",
            "--model_config",
            str(CONFIG_PATH),
            "--ckpt",
            str(checkpoint_path),
            "--video_path",
            str(input_path),
            "--reprojected_path",
            str(reprojected_path),
            "--reprojected_mask_path",
            str(reprojected_mask_path),
            "--output_folder",
            str(OUTPUT_ROOT),
            "--save_sbs",
            "--chunk_size",
            str(int(chunk_size)),
        ]
    )

    sbs_path = find_output_video(base_name, "sbs", started_at)
    if sbs_path is None:
        raise gr.Error("SBS動画が見つかりませんでした。")

    progress(1.0, desc="完了")
    return str(sbs_path), str(sbs_path)


def clean_outputs() -> str:
    """outputsフォルダの中身を削除する。"""
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for child in OUTPUT_ROOT.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
    return "outputsフォルダを掃除しました。"


with gr.Blocks(title="M2SVID") as demo:
    gr.Markdown("# M2SVID")
    with gr.Row():
        with gr.Column():
            video_input = gr.Video(label="Input video")
            chunk_size = gr.Slider(
                label="chunk_size",
                minimum=4,
                maximum=25,
                value=10,
                step=1,
            )
            checkpoint = gr.Radio(
                label="Checkpoint",
                choices=["fp16", "int8"],
                value="fp16",
            )
            run_button = gr.Button("Generate", variant="primary")
            clean_button = gr.Button("Clean outputs")
            clean_status = gr.Textbox(label="Status", interactive=False)
        with gr.Column():
            video_output = gr.Video(label="Generated video")
            download_output = gr.File(label="Download")

    run_button.click(
        fn=convert_video,
        inputs=[video_input, chunk_size, checkpoint],
        outputs=[video_output, download_output],
    )
    clean_button.click(fn=clean_outputs, outputs=clean_status)


if __name__ == "__main__":
    demo.queue().launch()
