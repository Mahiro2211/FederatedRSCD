#!/bin/bash

# 检查CUDA
echo "检查CUDA环境..."
python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}')"

# 检查依赖包
echo "检查依赖包..."
python -c "import numpy, torch, sklearn, matplotlib, loguru, wandb, tqdm, rich" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✅ 所有依赖包已安装"
else
    echo "⚠️  部分依赖包缺失，尝试安装..."
    pip install numpy torch scikit-learn matplotlib loguru wandb tqdm rich -q
fi

# 默认参数
BATCH_SIZE=${BATCH_SIZE:-8}
NUM_EPOCHS=${NUM_EPOCHS:-2}             # 测试时改为2轮
NUM_CLIENT_EPOCH=${NUM_CLIENT_EPOCH:-2} # 测试时改为2轮
LR=${LR:-0.0001}
FRAC=${FRAC:-0.5}
USE_PARALLEL=${USE_PARALLEL:-0}                     # 1表示使用，0表示不使用（测试时禁用）
N_WORKERS=${N_WORKERS:-2}                           # 测试时改为2个进程
NUM_WORKERS_DATALOADER=${NUM_WORKERS_DATALOADER:-0} # 测试时改为0（避免嵌套多进程）
SAVE_DIR=${SAVE_DIR:-./saved_models_test}

echo ""
echo "============================================================"
echo "  联邦学习训练 - 快速启动"
echo "============================================================"
echo ""
echo "训练配置:"
echo "  Batch Size: $BATCH_SIZE"
echo "  训练轮数: $NUM_EPOCHS"
echo "  客户端本地训练轮数: $NUM_CLIENT_EPOCH"
echo "  学习率: $LR"
echo "  客户端参与比例: $FRAC"

if [ "$USE_PARALLEL" -eq 1 ]; then
    echo "  使用并行训练: 是"
    PARALLEL_FLAG="--use_parallel"
else
    echo "  使用并行训练: 否"
    PARALLEL_FLAG=""
fi

echo "  并行进程数: $N_WORKERS"
echo "  数据加载器进程数: $NUM_WORKERS_DATALOADER"
echo "  保存目录: $SAVE_DIR"
echo ""
echo "============================================================"
echo ""

# 创建保存目录
mkdir -p "$SAVE_DIR"

# 开始训练
echo "开始训练..."

# 使用后台运行并记录PID
python main.py \
    --batch_size $BATCH_SIZE \
    --num_epochs $NUM_EPOCHS \
    --num_client_epoch $NUM_CLIENT_EPOCH \
    --lr $LR \
    --frac $FRAC \
    $PARALLEL_FLAG \
    --n_workers $N_WORKERS \
    --num_workers_dataloader $NUM_WORKERS_DATALOADER \
    --save_dir "$SAVE_DIR" &

TRAIN_PID=$!
echo "训练进程PID: $TRAIN_PID"
echo "查看完整日志: tail -f training.log"

# 等待测试完成后杀死进程
echo ""
echo "等待测试运行（30秒后自动停止）..."
sleep 30

# 检查进程是否还在运行
if ps -p $TRAIN_PID >/dev/null; then
    echo ""
    echo "测试运行正常，正在停止训练进程..."
    kill $TRAIN_PID
    echo "✅ 训练进程已停止"
else
    echo ""
    echo "训练进程已自动完成或停止"
fi

echo ""
echo "============================================================"
echo "  测试完成！"
echo "============================================================"
echo "结果保存在: $SAVE_DIR"
echo "============================================================"

# 退出conda环境
conda deactivate
