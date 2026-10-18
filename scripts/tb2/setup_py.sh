#!/bin/bash
set -x
export UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
export UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple
export UV_HTTP_TIMEOUT=300
UV=/opt/conda/bin/uv
cd /root/gpufree-data/tb2
# model download (hf-mirror)
( export HF_ENDPOINT=https://hf-mirror.com HF_HUB_ENABLE_HF_TRANSFER=0
  $UV venv -p /usr/bin/python3.12 /root/gpufree-data/tb2/hfenv && \
  $UV pip install -p /root/gpufree-data/tb2/hfenv/bin/python "huggingface_hub[cli]" && \
  /root/gpufree-data/tb2/hfenv/bin/hf download Qwen/Qwen3-8B --revision b968826d9c46dd6066d109eabc6255188de91218 --local-dir /root/gpufree-data/models/Qwen3-8B && echo MODEL_DONE ) > dl_model.log 2>&1 &
# vllm env
( $UV venv -p /usr/bin/python3.12 /root/gpufree-data/tb2/vllmenv && \
  $UV pip install -p /root/gpufree-data/tb2/vllmenv/bin/python vllm && echo VLLM_DONE ) > vllm_install.log 2>&1 &
# harbor
( $UV tool install --python /usr/bin/python3.12 harbor && echo HARBOR_DONE ) > harbor_install.log 2>&1 &
wait
echo ALL_PY_DONE
