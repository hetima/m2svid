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
DEPTH_MODELS = {
    "DepthCrafter": "depthcrafter",
    "Video-Depth-Anything Small": "video-depth-anything-small",
    "Video-Depth-Anything Base": "video-depth-anything-base",
    "Video-Depth-Anything Large": "video-depth-anything-large",
}


def run_command(args: list[str]) -> None:
    """外部コマンドをリポジトリルート基準で実行する。"""
    subprocess.run(args, cwd=APP_DIR, check=True)


def find_output_video(base_name: str, suffix: str, extension: str, started_at: float) -> Path | None:
    """今回の実行で生成された動画を探す。"""
    candidates = sorted(
        OUTPUT_ROOT.glob(f"{base_name}_{suffix}*{extension}"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for path in candidates:
        if path.stat().st_mtime >= started_at:
            return path
    return candidates[0] if candidates else None


def convert_video(
    video_path: str | None,
    depth_model_label: str,
    chunk_size: int,
    checkpoint_kind: str,
    output_format: str,
    save_sbs: bool,
    progress=gr.Progress(),
):
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
    depth_model_id = DEPTH_MODELS[depth_model_label]
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    run_args = [
        sys.executable,
        "run.py",
        "--model_id",
        depth_model_id,
        "--video_path",
        str(input_path),
        "--disparity_perc",
        "0.1",
        "--mask_antialias",
        "0",
        "--model_config",
        str(CONFIG_PATH),
        "--ckpt",
        str(checkpoint_path),
        "--chunk_size",
        str(int(chunk_size)),
        "--output_folder",
        str(OUTPUT_ROOT),
    ]
    extension = ".mov" if output_format == "ProRes.mov" else ".mp4"
    if output_format == "ProRes.mov":
        run_args.append("--use_prores")
    run_args.append("--save_sbs" if save_sbs else "--no-save_sbs")

    progress(0.05, desc="M2SVIDを実行中")
    run_command(run_args)

    generated_path = find_output_video(base_name, "generated", extension, started_at)
    if generated_path is None:
        raise gr.Error("Generated動画が見つかりませんでした。")

    download_paths = [str(generated_path)]
    if save_sbs:
        sbs_path = find_output_video(base_name, "sbs", extension, started_at)
        if sbs_path is None:
            raise gr.Error("SBS動画が見つかりませんでした。")
        download_paths.append(str(sbs_path))

    progress(1.0, desc="完了")
    return str(generated_path), download_paths


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
            depth_model = gr.Radio(
                label="Depth model",
                choices=list(DEPTH_MODELS.keys()),
                value="DepthCrafter",
            )
            checkpoint = gr.Radio(
                label="Checkpoint",
                choices=["fp16", "int8"],
                value="fp16",
            )
            output_format = gr.Radio(
                label="Output format",
                choices=["x264.mp4", "ProRes.mov"],
                value="x264.mp4",
            )
            save_sbs = gr.Checkbox(label="Generate SBS", value=True)
            run_button = gr.Button("Generate", variant="primary")
            clean_button = gr.Button("Clean outputs")
            clean_status = gr.Textbox(label="Status", interactive=False)
        with gr.Column():
            video_output = gr.Video(label="Generated video")
            download_output = gr.File(label="Download", file_count="multiple")

    run_button.click(
        fn=convert_video,
        inputs=[video_input, depth_model, chunk_size, checkpoint, output_format, save_sbs],
        outputs=[video_output, download_output],
    )
    clean_button.click(fn=clean_outputs, outputs=clean_status)


if __name__ == "__main__":
    demo.queue().launch()
