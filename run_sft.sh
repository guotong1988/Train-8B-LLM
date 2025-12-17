#!/bin/bash

# SFT训练脚本
# 使用方法: bash run_sft.sh 或 ./run_sft.sh

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
LOG_FILE="${LOGS_DIR}/sft_train_${TIMESTAMP}.log"

# 训练参数配置（可根据需要修改）
MODEL_NAME="/data/Qwen3-8B-Base"
OUTPUT_DIR="/data/outputs-sft-from-base/${TIMESTAMP}"
DATASET=""  # 留空使用默认数据集
# 子集名称配置（支持多个子集，用空格分隔）
# 留空或不设置则使用所有默认子集（chinese_traditional, coig_pc, exam, finance, douban, human_value, logi_qa, ruozhiba, segmentfault, wiki, wikihow, xhs, zhihu）
# 示例: SUBSET_NAME="coig_pc exam finance"  # 只使用指定的子集
# 示例: SUBSET_NAME=""  # 使用所有默认子集
SUBSET_NAME=""

# JSONL文件路径配置（优先于DATASET，如果设置了LLM_JSONL，将使用该文件）
LLM_JSONL=""

TEXT_COLUMN="text"

# 训练超参数
LEARNING_RATE=2e-5
BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=8  # 进一步增加梯度累积步数以减少显存（从4增加到8）
NUM_EPOCHS=1
MAX_SEQ_LENGTH=128  # 减小序列长度以节省显存（已改为256）
WARMUP_STEPS=100
SEED=42

# 训练配置
SAVE_STEPS=100000  # 增加保存间隔以减少I/O和显存峰值
LOGGING_STEPS=10
EVAL_STRATEGY="no"  # 可选: no, steps, epoch
EVAL_STEPS=500
SAVE_TOTAL_LIMIT=1  # 只保留1个检查点以节省磁盘和显存

# 性能优化选项（取消注释以启用）
# USE_BF16="--bf16"  # 使用bfloat16精度（推荐用于A100等GPU）
USE_FP16="--fp16"  # 使用float16精度
GRADIENT_CHECKPOINTING="--gradient_checkpointing"  # 启用梯度检查点以节省内存（重要：可节省约40-50%显存）
USE_8BIT_OPTIMIZER="--use_8bit_optimizer"  # 使用8-bit优化器（重要：可节省约50-75%优化器状态显存，需要安装bitsandbytes）
# PACKING="--packing"  # 启用序列打包以提高效率（会增加显存，暂时禁用）
# PIN_MEMORY="--dataloader_pin_memory"  # 启用数据加载器pin memory（会增加显存，暂时禁用）

# 其他选项
DATALOADER_NUM_WORKERS=0  # 减少数据加载器工作进程数以节省显存（从4减少到0）
MAX_STEPS=-1  # -1表示使用num_train_epochs

export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
# 分布式训练配置
# 如果仍然OOM，可以尝试减少GPU数量，例如改为2或1
NPROC_PER_NODE=8  # 每个节点的进程数（GPU数量）

# NCCL环境变量配置（解决通信超时问题）
export NCCL_DEBUG=INFO  # 设置为INFO以获取更多调试信息，如果问题解决后可以改为WARN
export NCCL_TIMEOUT=7200  # 增加超时时间到7200秒（2小时），默认是1800秒（30分钟）
export NCCL_IB_DISABLE=1  # 禁用InfiniBand（单机训练通常不需要）
export NCCL_P2P_DISABLE=0  # 启用P2P通信（GPU间直接通信）
export NCCL_SOCKET_IFNAME=^docker0,lo  # 排除docker和loopback接口
export NCCL_BLOCKING_WAIT=1  # 使用阻塞等待模式，更稳定
export NCCL_ASYNC_ERROR_HANDLING=1  # 启用异步错误处理，有助于诊断问题
export NCCL_TREE_THRESHOLD=0  # 强制使用ring算法，避免tree算法可能的问题
# 如果使用多机训练，可能需要设置：
# export NCCL_SOCKET_IFNAME=eth0  # 替换为实际的网络接口名称
# export NCCL_IB_DISABLE=0  # 如果使用InfiniBand，启用它

# Tokenization和数据处理优化（解决tokenization超时问题）
export TOKENIZERS_PARALLELISM=false  # 禁用tokenizer的多进程，避免与分布式训练冲突
export OMP_NUM_THREADS=1  # 限制OpenMP线程数，避免过度并行化
export MKL_NUM_THREADS=1  # 限制MKL线程数
# 这些设置可以避免在数据预处理阶段产生额外的进程竞争，确保所有分布式进程同步

# PyTorch分布式训练超时配置（确保与NCCL_TIMEOUT一致）
export TORCH_DISTRIBUTED_TIMEOUT=7200  # PyTorch分布式操作的超时时间（秒），与NCCL_TIMEOUT保持一致

# 构建命令
CMD="/opt/conda/envs/python3.10.13/bin/torchrun --standalone --nproc_per_node=${NPROC_PER_NODE} train_sft.py \
    --model_name ${MODEL_NAME} \
    --output_dir ${OUTPUT_DIR} \
    --learning_rate ${LEARNING_RATE} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
    --num_train_epochs ${NUM_EPOCHS} \
    --max_seq_length ${MAX_SEQ_LENGTH} \
    --warmup_steps ${WARMUP_STEPS} \
    --seed ${SEED} \
    --save_steps ${SAVE_STEPS} \
    --logging_steps ${LOGGING_STEPS} \
    --eval_strategy ${EVAL_STRATEGY} \
    --dataloader_num_workers ${DATALOADER_NUM_WORKERS} \
    --max_steps ${MAX_STEPS} \
    --save_total_limit ${SAVE_TOTAL_LIMIT}"

# 添加可选参数
# 优先使用LLM_JSONL文件（如果存在且设置了）
if [ -n "${LLM_JSONL}" ]; then
    if [ -f "${LLM_JSONL}" ]; then
        echo "检测到LLM_JSONL文件: ${LLM_JSONL}"
        CMD="${CMD} --dataset ${LLM_JSONL}"
    else
        echo "警告: LLM_JSONL文件不存在: ${LLM_JSONL}，将使用其他数据集配置"
        if [ -n "${DATASET}" ]; then
            CMD="${CMD} --dataset ${DATASET}"
        fi
    fi
elif [ -n "${DATASET}" ]; then
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

if [ -n "${TEXT_COLUMN}" ]; then
    CMD="${CMD} --text_column ${TEXT_COLUMN}"
fi

if [ -n "${USE_BF16}" ]; then
    CMD="${CMD} ${USE_BF16}"
fi

if [ -n "${USE_FP16}" ]; then
    CMD="${CMD} ${USE_FP16}"
fi

if [ -n "${GRADIENT_CHECKPOINTING}" ]; then
    CMD="${CMD} ${GRADIENT_CHECKPOINTING}"
fi

if [ -n "${USE_8BIT_OPTIMIZER}" ]; then
    CMD="${CMD} ${USE_8BIT_OPTIMIZER}"
fi

if [ -n "${PACKING}" ]; then
    CMD="${CMD} ${PACKING}"
fi

if [ -n "${PIN_MEMORY}" ]; then
    CMD="${CMD} ${PIN_MEMORY}"
fi

if [ "${EVAL_STRATEGY}" != "no" ]; then
    CMD="${CMD} --eval_steps ${EVAL_STEPS}"
fi

# 打印配置信息（同时输出到终端和日志文件）
{
    echo "=========================================="
    echo "SFT训练配置"
    echo "=========================================="
    echo "模型路径: ${MODEL_NAME}"
    echo "输出目录: ${OUTPUT_DIR}"
    if [ -n "${LLM_JSONL}" ] && [ -f "${LLM_JSONL}" ]; then
        echo "数据集: ${LLM_JSONL} (JSONL文件)"
    elif [ -n "${DATASET}" ]; then
        echo "数据集: ${DATASET}"
    else
        echo "数据集: 默认数据集 (AI-ModelScope/COIG-CQIA)"
    fi
    if [ -n "${SUBSET_NAME}" ]; then
        echo "子集名称: ${SUBSET_NAME}"
    else
        echo "子集名称: 所有默认子集 (chinese_traditional, coig_pc, exam, finance, douban, human_value, logi_qa, ruozhiba, segmentfault, wiki, wikihow, xhs, zhihu)"
    fi
    echo "学习率: ${LEARNING_RATE}"
    echo "批次大小: ${BATCH_SIZE}"
    echo "梯度累积步数: ${GRADIENT_ACCUMULATION_STEPS} (有效批次大小: $((BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * NPROC_PER_NODE)))"
    echo "训练轮数: ${NUM_EPOCHS}"
    echo "最大序列长度: ${MAX_SEQ_LENGTH}"
    echo "保存检查点限制: ${SAVE_TOTAL_LIMIT}"
    if [ -n "${USE_8BIT_OPTIMIZER}" ]; then
        echo "8-bit优化器: 已启用（可节省约50-75%优化器状态显存）"
    fi
    echo "随机种子: ${SEED}"
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
if [ ! -f "train_sft.py" ]; then
    echo "错误: 未找到train_sft.py文件"
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


