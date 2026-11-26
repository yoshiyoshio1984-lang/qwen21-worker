"""ビルド時にモデルをイメージへ同梱するスクリプト（Dockerfile から実行）。

- 無検閲トランスフォーマ（FP8, 約7.1GB）: chatrlabs の該当サブフォルダのみ
- ベース一式（テキストエンコーダ約17.5GB、VAE 等）: Qwen/Qwen-Image-2.1 から
  transformer フォルダを除いて取得（トランスフォーマは上記に差し替えるため）
合計 約26GB が HF_HOME（/opt/hf-cache）に入る。
実行時は HF_HUB_OFFLINE=1 でこのキャッシュだけを使うので、
ワーカー起動時のモデルダウンロードは発生しない。
"""

from huggingface_hub import snapshot_download

snapshot_download(
    "chatrlabs/Qwen-Image-2.1-Uncensored-Omni-Quantized",
    allow_patterns=["qwen-image-2.1-step90000-FP8-torchao/*"],
)
snapshot_download(
    "Qwen/Qwen-Image-2.1",
    ignore_patterns=["transformer/*"],
)
print("Baked models into HF cache.")
