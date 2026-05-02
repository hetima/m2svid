# merge_checkpoint.py

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

import torch
from omegaconf import OmegaConf
from sgm.util import instantiate_from_config
from safetensors.torch import save_file


def convert_precision(state_dict, dtype_str="fp16"):
    """精度変換。整数テンソルや小さなテンソルはそのまま"""
    new_sd = {}
    for k, v in state_dict.items():
        # 整数型やbool、スカラーはスキップ
        if not v.is_floating_point() or v.numel() <= 1:
            new_sd[k] = v
            continue

        if dtype_str == "fp32":
            new_sd[k] = v.float()
        elif dtype_str == "fp16":
            new_sd[k] = v.half()
        elif dtype_str == "bf16":
            new_sd[k] = v.bfloat16()
        elif dtype_str == "fp8_e4m3":
            new_sd[k] = v.to(torch.float8_e4m3fn)
        elif dtype_str == "fp8_e5m2":
            new_sd[k] = v.to(torch.float8_e5m2)
        elif dtype_str == "int8":
            # 対称的 per-tensor 量子化（ディスク容量削減のみ。GPU上ではfp16に戻る）
            # scale = max(|x|) / 127, q = round(x / scale).to(int8)
            abs_max = v.abs().max().clamp(min=1e-8)
            scale = abs_max / 127.0
            q = (v / scale).round().clamp(-128, 127).to(torch.int8)
            new_sd[k] = q
            new_sd[k + "_scale"] = scale.float()  # スケール因子を別キーで保存
        else:
            raise ValueError(f"Unknown dtype: {dtype_str}")
    return new_sd


def quantize_with_quanto(model, weights):
    """optimum-quantoでモデルを量子化してstate_dictを返す。
    
    Linear, Conv2d, LayerNormが量子化対象。
    Conv3d等の非対応レイヤー、およびfirst_stage_model (VAE) は除外。
    first_stage_modelのConv2dはtimesteps kwargを使うためQConv2dと非互換。
    量子化されたモデルはGPU上でもint8のまま保持されるため、
    実行時のGPUメモリ消費が削減される。
    """
    from optimum.quanto import quantize, freeze
    quantize(model, weights=weights, exclude=["first_stage_model*"])
    freeze(model)
    return model.state_dict()


# 使いたい精度を選択:
# "fp32", "fp16", "bf16", "fp8_e4m3", "fp8_e5m2", "int8", "quanto_int8"
precision = "quanto_int8"

if __name__ == "__main__":
    config = OmegaConf.load("configs/m2svid.yaml")
    model = instantiate_from_config(config.model)
    model.init_from_ckpt("ckpts/m2svid_weights.pt")
    model = model.half()  # fp16に統一してから量子化

    if precision == "quanto_int8":
        # optimum-quantoによる量子化（GPU上でint8のまま保持）
        from optimum.quanto import qint8
        converted_sd = quantize_with_quanto(model, weights=qint8)
    else:
        combined_sd = model.state_dict()
        converted_sd = convert_precision(combined_sd, precision)

    # サイズ計算
    new_size = sum(v.numel() * v.element_size() for v in converted_sd.values())
    print(f"Converted ({precision}): {new_size / 1e9:.2f} GB")
    print(f"Total keys: {len(converted_sd)}")

    # int8データキーの数を確認（quanto_int8の場合）
    if precision == "quanto_int8":
        data_keys = sum(1 for k in converted_sd if "._data" in k)
        scale_keys = sum(1 for k in converted_sd if "._scale" in k)
        other_keys = len(converted_sd) - data_keys - scale_keys
        print(f"  Quantized weight._data: {data_keys}")
        print(f"  Quantized weight._scale: {scale_keys}")
        print(f"  Other (fp16/fp32): {other_keys}")

    save_file(converted_sd, f"ckpts/m2svid_combined_{precision}.safetensors")
    print(f"Saved to ckpts/m2svid_combined_{precision}.safetensors")
