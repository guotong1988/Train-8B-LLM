#!/bin/bash

# PPO训练脚本
# 使用方法: bash run_ppo.sh 或 ./run_ppo.sh

# 设置脚本在遇到错误时退出
set -e

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 创建日志目录
LOGS_DIR=".logs"
mkdir -p "${LOGS_DIR}"

# 生成日志文件名（带时间戳）
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOGS_DIR}/ppo_train_${TIMESTAMP}.log"

# 训练参数配置（可根据需要修改）
MODEL_NAME="/data/outputs-sft"
REWARD_MODEL_NAME="/data/Skywork-Reward-V2-Qwen3-4B"
OUTPUT_DIR="/data/outputs-ppo"
DATASET=""  # 留空使用默认数据集 AI-ModelScope/COIG-CQIA
# 子集名称配置（支持多个子集，用空格分隔）
# 留空或不设置则使用所有默认子集（chinese_traditional, coig_pc, exam, finance, douban, human_value, logi_qa, ruozhiba, segmentfault, wiki, wikihow, xhs, zhihu）
# 示例: SUBSET_NAME="coig_pc exam finance"  # 只使用指定的子集
# 示例: SUBSET_NAME=""  # 使用所有默认子集
SUBSET_NAME=""
PROMPT_COLUMN="prompt"

# 训练超参数
LEARNING_RATE=5e-6
BATCH_SIZE=2
GRADIENT_ACCUMULATION_STEPS=1
NUM_EPOCHS=1
NUM_GENERATIONS=1  # Samples per prompt for PPO
SEED=42
MAX_STEPS=-1  # Override total training steps when dataset has no length
MAX_SEQ_LENGTH=1024  # 最大序列长度，用于对prompt进行截断，参考SFT脚本

# PPO特定参数
VF_COEF=0.1  # Value function loss coefficient
CLIPRANGE=0.2  # PPO clip range
CLIPRANGE_VALUE=0.2  # PPO value clip range

# 训练配置
SAVE_STEPS=1000
LOGGING_STEPS=10

# 性能优化选项（取消注释以启用）
# USE_BF16="--bf16"  # 使用bfloat16精度（推荐用于A100等GPU）
# USE_FP16="--fp16"  # 注意：目前脚本未直接使用fp16参数，保留占位
# PIN_MEMORY="--dataloader_pin_memory"  # 启用数据加载器pin memory
# 使用8-bit优化器（与 train_sft.py 保持一致，需要安装bitsandbytes）
USE_8BIT_OPTIMIZER="--use_8bit_optimizer"

# 其他选项
DATALOADER_NUM_WORKERS=1
PUSH_TO_HUB=""  # 如需推送到hub，设置为 "--push_to_hub"

export CUDA_VISIBLE_DEVICES='0,1'
# 分布式训练配置
# 如果仍然OOM，可以尝试减少GPU数量，例如改为2或1
NPROC_PER_NODE=2  # 每个节点的进程数（GPU数量）

# 构建命令
CMD="/opt/conda/envs/python3.10.13/bin/torchrun --standalone --nproc_per_node=${NPROC_PER_NODE} train_ppo.py \
    --model_name ${MODEL_NAME} \
    --reward_model_name ${REWARD_MODEL_NAME} \
    --output_dir ${OUTPUT_DIR} \
    --prompt_column ${PROMPT_COLUMN} \
    --learning_rate ${LEARNING_RATE} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
    --num_train_epochs ${NUM_EPOCHS} \
    --num_generations ${NUM_GENERATIONS} \
    --seed ${SEED} \
    --vf_coef ${VF_COEF} \
    --cliprange ${CLIPRANGE} \
    --cliprange_value ${CLIPRANGE_VALUE} \
    --dataloader_num_workers ${DATALOADER_NUM_WORKERS} \
    --max_steps ${MAX_STEPS} \
    --max_seq_length ${MAX_SEQ_LENGTH} \
    --save_steps ${SAVE_STEPS} \
    --logging_steps ${LOGGING_STEPS}"

# 添加可选参数
if [ -n "${DATASET}" ]; then
    CMD="${CMD} --dataset ${DATASET}"
fi

# 处理子集名称（支持多个子集）
if [ -n "${SUBSET_NAME}" ]; then
    # 将多个子集名称作为多个参数传递
    CMD="${CMD} --subset_name"
    for subset in ${SUBSET_NAME}; do
        CMD="${CMD} ${subset}"
    done
fi

# 添加可选参数
if [ -n "${USE_BF16}" ]; then
    CMD="${CMD} ${USE_BF16}"
fi

if [ -n "${USE_FP16}" ]; then
    CMD="${CMD} ${USE_FP16}"
fi

if [ -n "${PIN_MEMORY}" ]; then
    CMD="${CMD} ${PIN_MEMORY}"
fi

if [ -n "${USE_8BIT_OPTIMIZER}" ]; then
    CMD="${CMD} ${USE_8BIT_OPTIMIZER}"
fi

if [ -n "${PUSH_TO_HUB}" ]; then
    CMD="${CMD} ${PUSH_TO_HUB}"
fi

# 打印配置信息（同时输出到终端和日志文件）
{
    echo "=========================================="
    echo "PPO训练配置"
    echo "=========================================="
    echo "模型路径: ${MODEL_NAME}"
    echo "奖励模型路径: ${REWARD_MODEL_NAME}"
    echo "输出目录: ${OUTPUT_DIR}"
    echo "数据集: ${DATASET:-默认数据集 (AI-ModelScope/COIG-CQIA)}"
    if [ -n "${SUBSET_NAME}" ]; then
        echo "子集名称: ${SUBSET_NAME}"
    else
        echo "子集名称: 所有默认子集 (chinese_traditional, coig_pc, exam, finance, douban, human_value, logi_qa, ruozhiba, segmentfault, wiki, wikihow, xhs, zhihu)"
    fi
    echo "提示词列名: ${PROMPT_COLUMN}"
    echo "学习率: ${LEARNING_RATE}"
    echo "批次大小: ${BATCH_SIZE}"
    echo "梯度累积步数: ${GRADIENT_ACCUMULATION_STEPS} (有效批次大小: $((BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * NPROC_PER_NODE)))"
    echo "训练轮数: ${NUM_EPOCHS}"
    echo "每个提示词生成数量: ${NUM_GENERATIONS}"
    echo "最大训练步数: ${MAX_STEPS}"
    echo "最大序列长度: ${MAX_SEQ_LENGTH}"
    echo "价值函数系数: ${VF_COEF}"
    echo "PPO裁剪范围: ${CLIPRANGE}"
    echo "价值裁剪范围: ${CLIPRANGE_VALUE}"
    echo "保存步数: ${SAVE_STEPS}"
    echo "日志记录步数: ${LOGGING_STEPS}"
    echo "随机种子: ${SEED}"
    if [ -n "${USE_8BIT_OPTIMIZER}" ]; then
        echo "8-bit优化器: 已启用（可节省约50-75%优化器状态显存）"
    fi
    echo "GPU数量: ${NPROC_PER_NODE}"
    echo "日志文件: ${LOG_FILE}"
    echo "=========================================="
    echo ""
} | tee -a "${LOG_FILE}"

# 检查Python是否可用
if ! command -v python &> /dev/null; then
    echo "错误: 未找到python命令"
    exit 1
fi

# 检查训练脚本是否存在
if [ ! -f "train_ppo.py" ]; then
    echo "错误: 未找到train_ppo.py文件"
    exit 1
fi

# 执行训练命令（后台运行）
echo "开始执行训练..."
echo "命令: ${CMD}"
echo "日志文件: ${LOG_FILE}"
echo ""

# 使用 nohup 在后台运行，并将输出追加到日志文件
nohup bash -c "${CMD}" >> "${LOG_FILE}" 2>&1 &
TRAIN_PID=$!

# 等待一下确保进程启动
sleep 1

# 检查进程是否还在运行
if ps -p ${TRAIN_PID} > /dev/null 2>&1; then
    echo "训练已在后台启动！"
    echo "进程ID: ${TRAIN_PID}"
    echo "日志文件: ${LOG_FILE}"
    echo ""
    echo "查看日志: tail -f ${LOG_FILE}"
    echo "查看进程: ps -p ${TRAIN_PID}"
    echo ""
else
    echo "错误: 训练进程启动失败，请查看日志文件: ${LOG_FILE}"
    exit 1
fi
