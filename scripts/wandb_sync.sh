#!/bin/bash

# 设置wandb目录的路径，你可以根据你的wandb目录位置来修改这个路径
WANDB_DIR="./wandb"

# 遍历wandb目录下所有以offline-run-开头的目录
for dir in "$WANDB_DIR"/offline-run-*; do
  # 检查目录是否存在
  if [ -d "$dir" ]; then
    echo "Syncing $dir"
    # 使用wandb sync命令同步目录
    wandb sync "$dir" -p mm-mamba
  fi
done

echo "All offline runs have been synced."