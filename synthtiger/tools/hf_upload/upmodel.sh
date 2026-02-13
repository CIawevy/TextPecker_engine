#!/bin/bash

# 设置代理（和 updata.sh 保持一致）
export HTTP_PROXY=http://sys-proxy-rd-relay.byted.org:8118
export http_proxy=http://sys-proxy-rd-relay.byted.org:8118
export https_proxy=http://sys-proxy-rd-relay.byted.org:8118

# 设置 HF token（注意不要对外泄露此文件）
export HF_TOKEN=HF_TOKEN
export HUGGINGFACE_HUB_TOKEN=HF_TOKEN

# 激活 conda 环境
source /mnt/bn/ocr-generation-lf/zhuhanshen/home/anaconda3/etc/profile.d/conda.sh
conda activate flux

# 切换到工作目录
cd /mnt/bn/ocr-generation-lf/zhuhanshen/project/ms-swift

############################################
# 1) 上传 IT-SFT-MERGE-1128 的 checkpoint
############################################

# python upload_model.py \
#   --model_dir /mnt/bn/ocr-generation-lf/zhuhanshen/wuxc/ms-swift/IT-SFT-MERGE-1128/v2-20251128-180713/checkpoint-45850 \
#   --repo_id CIawevy/TextPecker-8B-InternVL3 \
#   --private \
#   --revision main \
#   --commit_message "Upload TextPecker-8B-InternVL3"

############################################
# 2) 上传 sft-qwen3vl-MERGE-1202 的 checkpoint
############################################

# python upload_model.py \
#   --model_dir /mnt/bn/ocr-generation-lf/zhuhanshen/project/ms-swift/sft-qwen3vl-MERGE-1202/v0-20251202-110143/checkpoint-45850 \
#   --repo_id CIawevy/TextPecker-8B-Qwen3VL \
#   --private \
#   --revision main \
#   --commit_message "Upload CIawevy/TextPecker-8B-Qwen3VL"

  python upload_model.py \
  --model_dir /mnt/bn/ocr-generation-lf/zhuhanshen/wuxc/ms-swift/TextPecker-8B-InternVL3_5/v0-20260112-014010/checkpoint-45000 \
  --repo_id CIawevy/TextPecker-8B-InternVL3_5  \
  --private \
  --revision main \
  --commit_message "Upload CIawevy/TextPecker-8B-InternVL3_5"

  
