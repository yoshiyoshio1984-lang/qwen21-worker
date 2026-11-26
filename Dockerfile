FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# モデル同梱は一時停止中（2026-11-27）: 同梱イメージ（30.47GB）では
# ワーカーが exit code 127 で無言終了する問題が未解決のため、まず
# 通常イメージで生成を成立させる。bake_models.py はリポジトリに残してある。

COPY handler.py .
CMD ["python", "-u", "handler.py"
