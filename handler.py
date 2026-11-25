# Qwen-Image-2.1 (Uncensored Omni, FP8) ワーカー
# - 画像生成と画像編集を同じパイプラインで行う統合モデル
# - ベース Qwen/Qwen-Image-2.1（text encoder Qwen3-VL-8B + VAE + scheduler）
# - transformer は chatrlabs の Uncensored Omni FP8 版（7.1GB）
# - diffusers 0.41.0 以降の QwenImage21Pipeline が必要
#   （0.40.0 には QwenImage21Pipeline が無いので 0.41.0 を指定）
# - ライセンスは qwen-research（非商用）。個人利用の範囲で使うこと。
# - HF_TOKEN は不要（Qwen-Image-2.1 は gated ではない）
# - 2509 ワーカーの失敗（transformers 4.49.0 + Qwen2.5-VL の
#   decoder_config.to_dict() エラー）とは別系統。
#   こちらは Qwen3-VL-8B を使うので同じエラーは出ない見込み。
#   transformers は 4.x の最新（4.57.6）に固定し、5.x は避ける。
import base64
import io
import os

import runpod
import torch
from diffusers import QwenImage21Pipeline, QwenImage21Transformer2DModel
from PIL import Image

BASE_MODEL_ID = os.environ.get("BASE_MODEL_ID", "Qwen/Qwen-Image-2.1")
TRANSFORMER_REPO = os.environ.get(
    "TRANSFORMER_REPO", "chatrlabs/Qwen-Image-2.1-Uncensored-Omni-Quantized"
)
TRANSFORMER_SUBFOLDER = os.environ.get(
    "TRANSFORMER_SUBFOLDER", "qwen-image-2.1-step90000-FP8-torchao"
)

_pipe = None


def _get_pipe():
    global _pipe
    if _pipe is None:
        print(f"Loading transformer {TRANSFORMER_REPO}/{TRANSFORMER_SUBFOLDER} ...", flush=True)
        transformer = QwenImage21Transformer2DModel.from_pretrained(
            TRANSFORMER_REPO,
            subfolder=TRANSFORMER_SUBFOLDER,
            torch_dtype=torch.bfloat16,
        )
        print(f"Loading base pipeline {BASE_MODEL_ID} ...", flush=True)
        _pipe = QwenImage21Pipeline.from_pretrained(
            BASE_MODEL_ID,
            transformer=transformer,
            torch_dtype=torch.bfloat16,
        )
        _pipe.enable_model_cpu_offload()
        print("Model loaded.", flush=True)
    return _pipe


def _round16(n):
    return max(16, int(round(n / 16.0)) * 16)


def handler(event):
    inp = event.get("input", {}) or {}
    prompt = inp.get("prompt", "")
    if not prompt:
        return {"error": "prompt is required"}

    # README の推奨：negative_prompt を空にすると true CFG が無効になるため、
    # 未指定時は半角スペース 1 つを送る。
    negative_prompt = inp.get("negative_prompt", None)
    if negative_prompt is None:
        negative_prompt = " "
    steps = int(inp.get("num_inference_steps", 35))

    # true_cfg_scale：README 推奨 5.0。アプリが送る guidance_scale があれば
    # それを使い、1.0〜5.0 に収める。ネガティブ無しなら 1.0。
    cfg = inp.get("true_cfg_scale", inp.get("guidance_scale", None))
    if cfg is None:
        cfg = 5.0 if (negative_prompt and negative_prompt.strip()) else 1.0
    cfg = max(1.0, min(5.0, float(cfg)))

    seed = inp.get("seed")
    generator = None
    if seed is not None:
        generator = torch.Generator("cuda").manual_seed(int(seed))
    else:
        seed = torch.randint(0, 2**31 - 1, (1,)).item()
        generator = torch.Generator("cuda").manual_seed(int(seed))

    pipe = _get_pipe()

    image_b64 = inp.get("image_base64") or inp.get("image")
    mode = "qwen21-txt2img"
    kwargs = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "num_inference_steps": steps,
        "true_cfg_scale": cfg,
        "use_kv_cache": True,
        "generator": generator,
    }
    if image_b64:
        mode = "qwen21-edit"
        raw = base64.b64decode(image_b64.split(",", 1)[-1])
        img = Image.open(io.BytesIO(raw)).convert("RGBA")
        w = _round16(inp.get("width", img.width))
        h = _round16(inp.get("height", img.height))
        kwargs["image"] = [img]
        kwargs["width"] = w
        kwargs["height"] = h
        kwargs["output_resolution"] = max(w, h)
    else:
        w = _round16(inp.get("width", 1024))
        h = _round16(inp.get("height", 1024))
        kwargs["width"] = w
        kwargs["height"] = h

    out = pipe(**kwargs)
    result = out.images[0].convert("RGB")
    buf = io.BytesIO()
    result.save(buf, format="PNG")
    return {
        "image_base64": base64.b64encode(buf.getvalue()).decode("utf-8"),
        "width": result.width,
        "height": result.height,
        "seed": int(seed),
        "model": f"{TRANSFORMER_REPO}/{TRANSFORMER_SUBFOLDER}",
        "mode": mode,
    }


runpod.serverless.start({"handler": handler})
