FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# モデル同梱（2026-11-26）: ビルド時にモデル一式（約26GB）をイメージ内へ取り込む。
# 実行時は HF_HUB_OFFLINE=1 でこのキャッシュだけを使い、ワーカー起動時の
# モデルダウンロード（課金対象）を発生させない。
ENV HF_HOME=/opt/hf-che
COPY bake_models.py .
RUN python bake_models.py && du -sh /opt/hf-cache
ENV HF_HUB_OFFLINE=1

COPY handler.py .
CMD ["python", "-u", "handler.py"
