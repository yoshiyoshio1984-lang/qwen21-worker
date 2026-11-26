"""ビルド時にモデルをイメージへ同梱するスクリプト（Dockerfile から実行）。

- 無検閲トランスフォーマ（FP8, 約7.1GB）: chatrlabs の該当サブフォルダのみ
- ベース一式（テキストエンコーダ約17.5GB、VAE 等）: Qwen/Qwen-Image-2.1 から
  transformer フォルダを除いて取得（トランスフォーマは上記に差し替えるため）
合計 約26GB が HF_HOME（/opt/hf-cache）に入る。
実行時は HF_HUB_OFFLINE=1 でこのキャッシュだけを使うので、
ワーカー起動時のモデルダウンロードは発生しない。

2026-11-27 修正: 前回のビルドではダウンロード後に /opt/hf-cache が
存在せず du で失敗したため、保存先を環境変数で明示し、
ダウンロード直後にこのスクリプト自身が中身を検証する。
"""

import os
import sys

# huggingface_hub を import する前に保存先を確定させる
os.environ["HF_HOME"] = "/opt/hf-cache"
os.environ["HF_HUB_CACHE"] = "/opt/hf-cache/hub"

from huggingface_hub import snapshot_download
from huggingface_hub.constants import HF_HOME, HF_HUB_CACHE

print(f"HF_HOME env: {os.environ.get('HF_HOME')}")
print(f"resolved HF_HOME: {HF_HOME}")
print(f"resolved HF_HUB_CACHE: {HF_HUB_CACHE}")

snapshot_download(
    "chatrlabs/Qwen-Image-2.1-Uncensored-Omni-Quantized",
    allow_patterns=["qwen-image-2.1-step90000-FP8-torchao/*"],
)
snapshot_download(
    "Qwen/Qwen-Image-2.1",
    ignore_patterns=["transformer/*"],
)

# 同梱できたか、その場で検証する
total = 0
for root, _dirs, files in os.walk("/opt/hf-cache"):
    for name in files:
        total += os.path.getsize(os.path.join(root, name))
print(f"Baked cache size: {total / 1e9:.1f} GB under /opt/hf-cache")
if total < 20e9:
    print("ERROR: baked cache is unexpectedly small", file=sys.stderr)
    sys.exit(1)
print("Baked models into HF cache.")
# rebuild after cache path fix 2026-11-27
