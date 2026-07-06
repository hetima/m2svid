# About this fork
- conda不要でvenvで使えるようにしたました（最低限の`requirements.txt`）
- なんか最終処理が25フレームまでしか対応してないみたいなので分割処理して結合するようにしたました（`inpaint_and_refine.py`）
- `m2svid_weights.pt`と`open_clip_pytorch_model.bin`を合体させた safetensors
- サブモジュールのリポジトリを直接内包
- Video-Depth-Anything実装
- Gradioインターフェイス
- 一発生成コマンド

## インストール

```
git clone https://github.com/hetima/m2svid.git
```

>Windows 11、python 3.12 の環境で動作確認しています。

torch関連は`requirements.txt`に書いてないので個別にインストールしてください。以下はpython 3.12のvenvにインストールする例です。flash_attnは [ussoewwin/Flash-Attention-2_for_Windows · Hugging Face](https://huggingface.co/ussoewwin/Flash-Attention-2_for_Windows) からダウンロードできます。GitHubとかにもあると思います。torch 2.10だとうまく動く組み合わせを見つけられませんでした。2.9.1が無難だと思います。2.9.1用のxformersは0.0.33.post2みたいですが、それだと動かなかったので0.3.3を入れました。

```sh
uv pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu130
uv pip install xformers==0.0.33 --no-deps #--no-depsしないと2.9が入る
uv pip install "path/to/flash_attn-2.8.3+cu130torch2.9.1cxx11abiTRUE-cp312-cp312-win_amd64.whl"

uv pip install -r requirements.txt
```

使い方はオリジナルと同じです。`PYTHONPATH`の追加はスクリプト内でするようにしたので不要です。gradioやすべての処理を一括で実行するスクリプトを追加しています。

必要なモデルは [Hugging Face](https://huggingface.co/hrktxz/m2svid_models) にまとめています。`ckpts`という名前のフォルダを作ってこれをそのまま配置すればOKです。

`inpaint_and_refine.py`に`--save_sbs`と`--save_anaglyph`のフラグを付けて生成するファイルを選べるようにしました。sbsはデフォルトで生成されます。オフにしたい場合`--no-save_sbs`を付けてください。

また、`--chunk_size`パラメータも付けました。一度に処理するフレーム数を指定できます（デフォルト10、最大25）。VRAM12GBで512x512の動画をそれなりの速度で処理できる限界は12くらいです。

`--use_prores`フラグも付けました。これを渡すとmp4ではなくProRes LTのmovを書き出します。

深度推定を[Video-Depth-Anything](https://github.com/DepthAnything/Video-Depth-Anything)にも対応しました。従来のDepthCrafterよりも圧倒的に速いです。`third_party/DepthCrafter/run.py` ではなく `get_depth.py` を使用し、`--model_id` 引数を設定してください。`depthcrafter`、`video-depth-anything-small`、`video-depth-anything-base`、`video-depth-anything-large` のいずれかです。モデルによってライセンスが異なるので本家を参照してください。800x600くらいならVRAM12GBでlarge行けます。

## m2svid_combined_quanto_int8.safetensors
`m2svid_weights.pt`と`open_clip_pytorch_model.bin`を合体させたものです。[Hugging Face](https://huggingface.co/hrktxz/m2svid_combined) からダウンロードできます。

optimum-quantoを使って一部パラメータをint8にしたものです。VRAM12GBでメモリ消費量はあんまり変わらない気がしますが、処理速度は速くなっています。`--model_config`に`m2svid_combined.yaml`を指定し、`--quanto_int8`フラグを付け加えて実行してください（ファイル名に quanto_int8 が含まれていたら自動判定するようにはしています）

```sh
python inpaint_and_refine.py  \
        --mask_antialias 0 \
        --model_config configs/m2svid_combined.yaml \
        --ckpt ckpts/m2svid_combined_quanto_int8.safetensors \
        --video_path demo/input.mp4  \
        --reprojected_path outputs/reprojected/input_reprojected.mp4 \
        --reprojected_mask_path outputs/reprojected/input_reprojected_mask.mp4\
        --output_folder outputs/m2svid \
        --quanto_int8 \
```


## m2svid_combined_fp16.safetensors

fp16 に変換したものです。処理速度やメモリ消費量はたぶん変わりません。あんまり意味ありません。ファイルサイズを削減するだけです。

`--model_config`に`m2svid_combined.yaml`を指定して使用してください。`--quanto_int8` フラグは付けないでください。


## run.py

`get_depth.py`、`warping.py`、`inpaint_and_refine.py`を順番に呼び出す一括実行用スクリプトです。基本的にはこれを使えば、深度推定、右目画像のreproject、inpaint/refineまでまとめて実行できます。

```sh
python run.py \
  --model_id video-depth-anything-base \
  --video_path path/to/input.mp4 \
  --disparity_perc 0.1 \
  --mask_antialias 0 \
  --model_config configs/m2svid_combined.yaml \
  --ckpt ckpts/m2svid_combined/m2svid_combined_fp16.safetensors \
  --use_prores \
  --no-save_sbs \
  --chunk_size 8 \
  --output_folder outputs
```

Windows PowerShellの場合は行末を `` ` `` にしてください。

```ps1
python run.py `
  --model_id video-depth-anything-base `
  --video_path "path\to\input.mp4" `
  --disparity_perc 0.1 `
  --mask_antialias 0 `
  --model_config "configs\m2svid_combined.yaml" `
  --ckpt "ckpts\m2svid_combined\m2svid_combined_fp16.safetensors" `
  --use_prores `
  --no-save_sbs `
  --chunk_size 8 `
  --output_folder "outputs"
```

`output_folder`の中に入力動画のstem名で作業フォルダを作ります。例えば`input.mp4`なら、途中ファイルは主に`outputs/input/`以下に保存されます。最終成果物は`outputs/input_generated.mov`のように`output_folder`直下へ保存されます。

途中ファイルが存在する場合は再利用します。`.npz`があれば深度推定をスキップし、`*_reprojected.mp4`と`*_reprojected_mask.mp4`があればwarpingをスキップします。inpaint/refineは毎回実行します。

入力動画の幅と高さが64の倍数である必要がありますが、そうでないファイルを渡した場合は、`outputs/<stem>/<stem>_padded.mov`をProRes LTで作成し、`outputs/<stem>/padding.json`に元サイズとpadding情報を保存します。以降の処理はpadded動画で実行し、最終的なgenerated/SBS出力だけ元のサイズへcropして戻します。

主な引数:

- `--model_id`: 深度推定モデル。`depthcrafter`、`video-depth-anything-small`、`video-depth-anything-base`、`video-depth-anything-large`などを指定します。
- `--video_path`: 入力動画のパス。
- `--disparity_perc`: 横幅に対する視差量の割合。例: `0.1`なら動画幅の10%を基準にreprojectします。
- `--mask_antialias`: reproject maskを縮小するときのantialias指定。既存の実行例では`0`。
- `--model_config`: M2SVIDのconfig yaml。通常は`configs/m2svid_combined.yaml`。
- `--ckpt`: M2SVIDのcheckpoint。fp16版なら`ckpts/m2svid_combined/m2svid_combined_fp16.safetensors`。
- `--use_prores`: 最終出力をmp4ではなくProRes LTの`.mov`で保存します。
- `--save_sbs` / `--no-save_sbs`: SBS動画を保存するかどうか。デフォルトは保存。
- `--chunk_size`: 一度にinpaint/refineするフレーム数。最大25。VRAMが足りない場合は小さくしてください。
- `--output_folder`: 作業フォルダと最終出力を置くフォルダ。


## Gradio

`app.py` でGradioサーバーが立ち上がります。必要なモデルは全部自動で取ってきます。


## CLIで使う例

対象動画ファイルだけ引数に渡して一発変換するPowerShellスクリプト例（リポジトリをカレントディレクトリにして実行してください）

```ps1
# conv filepath_to_convert [project_name]
function global:conv() {
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    # Set-Location "path/to/m2svid"

    # setting
    $outputPath = "outputs"
    $cnfg = "configs/m2svid_combined.yaml"
    # $ckpt = "ckpts/m2svid_combined/m2svid_combined_quanto_int8.safetensors"
    $ckpt = "ckpts/m2svid_combined/m2svid_combined_fp16.safetensors"
    # video-depth-anything-small video-depth-anything-base depthcrafter
    $deptModel = "video-depth-anything-large"

    if ($null -eq $args[0]) {
        Write-Output "no file path"
        return
    }
    if (!(Test-Path $args[0])) {
        Write-Output "file path does not exists"
        return
    }
    $path = $args[0]
    $fullPath = (Resolve-Path "$path").Path

    python run.py --model_id "$deptModel" --video_path "$fullPath" --output_folder "$outputPath" --disparity_perc 0.1 --mask_antialias 0 --model_config "$cnfg" --ckpt "$ckpt" --use_prores --no-save_sbs --chunk_size 10

    $sw.Stop()
    Write-Host "Elapsed Time: $($sw.Elapsed.ToString("hh\:mm\:ss"))" -ForegroundColor Cyan
}
```

以下オリジナルのREADME

# M2SVid: End-to-End Inpainting and Refinement for Monocular-to-Stereo Video Conversion


[![Project Page](https://img.shields.io/badge/Project%20Page-m2svid.github.io-blue?style=flat&logo=googlechrome)](https://m2svid.github.io/)
[![arXiv](https://img.shields.io/badge/arXiv-2505.16565-b31b1b.svg?style=flat&logo=arxiv)](https://arxiv.org/abs/2505.16565)


by [**Nina Shvetsova**](https://ninatu.github.io/), [**Goutam Bhat**](https://goutamgmb.github.io/), [**Prune Truong**](https://prunetruong.com/), [**Hilde Kuehne**](https://hildekuehne.github.io/), [**Federico Tombari**](https://federicotombari.github.io/)

**Accepted to 3DV 2026!**

<p align="center">
  <img src="teaser.gif" width="600">
</p>

**Update:** [March 20, 2026] We have released the pre-trained M2SVid weights! 

---

*This is not an officially supported Google product. This project is not
eligible for the [Google Open Source Software Vulnerability Rewards
Program](https://bughunters.google.com/open-source-security).*

---

</div>

## 📄 Abstract

We tackle the problem of monocular-to-stereo video conversion and propose a novel architecture for inpainting and refinement of the warped right view obtained by depth-based reprojection of the input left view.
We extend the Stable Video Diffusion (SVD) model to utilize the input left video, the warped right video, and the disocclusion masks as conditioning input to generate a high-quality right camera view. In order to effectively exploit information from neighboring frames for inpainting, we modify the attention layers in SVD to compute full attention for discoccluded pixels. Our model is trained to generate the right view video in an end-to-end manner without iterative diffusion steps by minimizing image space losses to ensure high-quality generation.
**Our approach outperforms previous state-of-the-art methods, being ranked best 2.6× more often than the second-place method in a user study, while being 6× faster.**



## 🛠️ Get started

### Weights

1. Download `ckpts.zip` from [Hi3D repo](https://github.com/yanghb22-fdu/Hi3D-Official) and unzip (follow step "2. Download checkpoints here and unzip."). Our model follows Hi3D implementation and uses the same openclip model.

2. Download the [M2SVid weights (8.5Gb)](https://storage.googleapis.com/gresearch/m2svid/m2svid_weights.zip) and extract them into the `ckpts` folder: `unzip m2svid_weights.zip -d ckpts/`. We provide two model variants: one with **full attention** for disoccluded tokens ([m2svid_weights.pt, 4.64Gb](https://storage.googleapis.com/gresearch/m2svid/m2svid_weights.pt)) and one **without full attention** ([m2svid_no_full_atten_weights.pt, 4.6Gb](https://storage.googleapis.com/gresearch/m2svid/m2svid_no_full_atten_weights.pt)). 

3. Optional (for training only): download [stable-video-diffusion-img2vid-xt checkpoint](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt) and put it in `ckpts/`.


### Environment

1. Create conda env `depthcrafter` following [DepthCrafter instructions](https://github.com/Tencent/DepthCrafter)
2. Create conda env `sgm`. We used cuda 11.8, `python=3.10.6`, `torch==2.0.1 torchvision==0.15.2`. We tested our model training/inference on GPUs A100 and H100.

```bash
conda env create -f environment.yml -n sgm
```


## ⚙️ Inference

Run inference on demo video:

```bash
bash inference.sh
```

See examples outputs in `demo` folder.

**Note 1:** The width/hight of the video should be divisible by 64.

**Note 2:** The model was trained on a resolution of 512x512. For inference of higher resolution videos, please follow the tiling approach described in the [StereoCrafter paper](https://stereocrafter.github.io/). Our released models support temporal and spatial stitching.

### Inference Steps:
1. **Depth prediction and depth-based warping**

```bash
source /opt/conda/bin/activate ""
conda activate depthcrafter
PYTHONPATH="third_party/DepthCrafter/::${PYTHONPATH}" python third_party/DepthCrafter/run.py  \
        --video-path demo/input.mp4 --save_folder outputs/depthcrafter --save_npz True --num_inference_steps 25 --max_res 1024

PYTHONPATH="./:./third_party/Hi3D_Official/:./third_party/pytorch_msssim/:${PYTHONPATH}" python warping.py  \
        --video_path demo/input.mp4 \
        --depth_path outputs/depthcrafter/input.npz \
        --output_path_reprojected outputs/reprojected/input_reprojected.mp4  \
        --output_path_mask outputs/reprojected/input_reprojected_mask.mp4 \
        --disparity_perc 0.05
```

2. **Inpainting and refinement with M2SVid**

```bash
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
```

**Note:** If you are using the version without full attention, ensure you use the `m2svid_no_full_atten.yaml` config instead:


```bash
source /opt/conda/bin/activate ""
conda activate sgm
PYTHONPATH="./:./third_party/Hi3D_Official/:./third_party/pytorch_msssim/:${PYTHONPATH}" python inpaint_and_refine.py  \
        --mask_antialias 0 \
        --model_config configs/m2svid_no_fullatten.yaml \
        --ckpt ckpts/m2svid_no_full_atten_weights.pt \
        --video_path demo/input.mp4  \
        --reprojected_path outputs/reprojected/input_reprojected.mp4 \
        --reprojected_mask_path outputs/reprojected/input_reprojected_mask.mp4\
        --output_folder outputs/m2svid_no_full_atten \
```



## 🏋️ Training and Quantitative Evaluation

### Datasets

We used the [Ego4D](https://ego4d-data.org/) and [Stereo4D](https://stereo4d.github.io/) datasets for model training and evaluation.

1. Download and preprocess the Stereo4D dataset into the folder `datasets/stereo4d` by following the [official instructions](https://github.com/Stereo4d/stereo4d-code). You only need to perform the rectification and stereo matching steps. Then, you can warp all videos using our `warping.py` script. At the end, you should have the following folders: `left_rectified`, `right_rectified`, `reprojected`, and `reprojected_mask`. We provide the train/val split in `datasets/stereo4d/subsets`.

2.  For Ego4D, we use only videos with the attribute `is_stereo=True`, resulting in 263 videos in total.  Download videos into `datasets/ego4d` by following the [official instructions](https://ego4d-data.org/). We rectify the videos, split them into 150-frames clips, and apply the BiDAStereo model to estimate disparities. Check the [**ego4d preprocessing README**](data_preprocess/) for more details.   At the end, you should have the following folders: `cropped_videos` (side by side rectified and cropped left and right videos), `reprojected`, and `reprojected_mask`. We provide the train/val split in `datasets/ego4d/subsets`.

### Training 

1. Download [stable-video-diffusion-img2vid-xt checkpoint](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt) and put it to ckpts. 

2. Run `make_m2svid_init.py` to modify SVD models weights for ours M2SVid model configuration with left view, warped view and mask conditioning. 

```bash
source /opt/conda/bin/activate ""
conda activate sgm
PYTHONPATH="./:./third_party/Hi3D_Official/:./third_party/pytorch_msssim/:${PYTHONPATH}" python make_m2svid_init.py
```

3. Run training 
```bash
source /opt/conda/bin/activate ""
conda activate sgm
PYTHONPATH="./:./third_party/Hi3D_Official/:./third_party/pytorch_msssim/:${PYTHONPATH}" python third_party/Hi3D_Official/train_test_updated.py \
    --base configs/training/m2svid_train.yaml \
    --no-test True \
    --train True \
    --logdir outputs/training/m2svid
```

### Evaluation

Evaluation on stereo4d: 

```bash
source /opt/conda/bin/activate ""
conda activate sgm
PYTHONPATH="./:./third_party/Hi3D_Official/:./third_party/pytorch_msssim/:${PYTHONPATH}" python third_party/Hi3D_Official/train_test_updated.py \
    --base configs/training/m2svid_train.yaml \
    --dataset_base configs/testing/stereo4d.yaml \
    --no-test False \
    --train False \
    --logdir outputs/training/m2svid \
    --resume /home/jupyter/outputs_m2svid/training/m2svid/checkpoints/epoch=000120.ckpt
```

Evaluation on ego4d:

```bash
source /opt/conda/bin/activate ""
conda activate sgm
PYTHONPATH="./:./third_party/Hi3D_Official/:./third_party/pytorch_msssim/:${PYTHONPATH}" python third_party/Hi3D_Official/train_test_updated.py \
    --base configs/training/m2svid_train.yaml \
    --dataset_base configs/testing/ego4d.yaml \
    --no-test False \
    --train False \d
    --logdir outputs/training/m2svid \
    --resume /home/jupyter/outputs_m2svid/training/m2svid/checkpoints/epoch=000000.ckpt
```

### Evaluation of Released Models

To reproduce the paper's results on **Stereo4D** and **Ego4D** using our released weights:

```bash
source /opt/conda/bin/activate ""
conda activate sgm

# Evaluate on Stereo4D
PYTHONPATH="./:./third_party/Hi3D_Official/:./third_party/pytorch_msssim/:${PYTHONPATH}" python third_party/Hi3D_Official/train_test_updated.py \
    --base configs/testing/pretrained_m2svid.yaml \
    --dataset_base configs/testing/stereo4d.yaml \
    --no-test False \
    --train False \
    --logdir outputs/training/m2svid 

# Evaluate on Ego4D
PYTHONPATH="./:./third_party/Hi3D_Official/:./third_party/pytorch_msssim/:${PYTHONPATH}" python third_party/Hi3D_Official/train_test_updated.py \
    --base configs/training/pretrained_m2svid.yaml \
    --dataset_base configs/testing/stereo4d.yaml \
    --no-test False \
    --train False \
    --logdir outputs/training/m2svid 
```

## 🎓 Citation

```bibtex
@article{shvetsova2026m2svid,
  title={M2SVid: End-to-End Inpainting and Refinement for Monocular-to-Stereo Video Conversion},
  author={Shvetsova, Nina and Bhat, Goutam and Truong, Prune and Kuehne, Hilde and Tombari, Federico},
  journal={3DV},
  year={2026}
}
```
