
export PATH="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtsearch-assistant/ai-search/zhangqi276/qzhang_test/env-verl/bin:$PATH"
which python3
which pip3

pip install math_verify

set -x

VLLM_USE_V1=1 python evaluate2.py 




# Shift the arguments so $@ refers to the rest


