#!/bin/bash

# 设置代理
export HTTP_PROXY=http://sys-proxy-rd-relay.byted.org:8118
export http_proxy=http://sys-proxy-rd-relay.byted.org:8118
export https_proxy=http://sys-proxy-rd-relay.byted.org:8118
export HF_TOKEN=HF_TOKEN
export HUGGINGFACE_HUB_TOKEN=HF_TOKEN
# 激活conda环境
source /mnt/bn/ocr-generation-lf/zhuhanshen/home/anaconda3/etc/profile.d/conda.sh
conda activate flux

LOG_FILE="/mnt/bn/ocr-generation-lf/zhuhanshen/project/log_updata.log"
mkdir -p "$(dirname "$LOG_FILE")"

exec > >(tee -a "$LOG_FILE") 2>&1

echo "===== $(date '+%F %T') [PID $$] updata.sh START ====="
trap 'exit_code=$?; echo "===== $(date '+%F %T') [PID $$] updata.sh END (exit=${exit_code}) ====="; exit $exit_code' EXIT

# 切换到工作目录
cd /mnt/bn/ocr-generation-lf/zhuhanshen/project/ms-swift


# 上传测试数据集
# python upload_dataset.py \
#   --input /mnt/bn/ocr-generation-lf/zhuhanshen/wuxc/ms-swift/data_our/JREC_EVALFIX_1.06k.json \
#   --repo_id CIawevy/TextPecker-1.5M \
#   --private \
#   --split test \


# 上传训练数据集
python upload_dataset.py \
  --input /mnt/bn/ocr-generation-lf/zhuhanshen/wuxc/ms-swift/data_our/JREC_MERGE_FIX_V3_1.48M.json \
  --repo_id CIawevy/TextPecker-1.5M \
  --split train \
  --private \
