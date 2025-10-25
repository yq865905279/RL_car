#!/bin/bash

# --- 智能训练启动脚本 ---
#
# 功能:
# 1. 使用指定的YAML配置文件启动单线程训练。
# 2. 自动将YAML配置转换为命令行参数。
# 3. 允许在命令行中覆盖配置文件中的参数。
#
# 使用说明:
#
# 1. 赋予执行权限:
#    chmod +x run_training.sh
#
# 2. 使用配置文件运行:
#    ./run_training.sh -c configs/base_config.yaml
#
# 3. 使用配置文件并覆盖特定参数:
#    ./run_training.sh -c configs/base_config.yaml --total_steps 5000 --learning_rate 1e-4
#
# --------------------------------------------------------------------------

set -e

# --- 参数解析 ---
CONFIG_FILE=""
OTHER_ARGS=()

while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -c|--config)
        CONFIG_FILE="$2"
        shift # past argument
        shift # past value
        ;;
        *)
        # 保存所有其他参数
        OTHER_ARGS+=("$1")
        shift # past argument
        ;;
    esac
done

# 检查是否提供了配置文件
if [ -z "$CONFIG_FILE" ]; then
    echo "错误: 未通过 -c 或 --config 提供配置文件路径。"
    echo "用法: $0 -c <config_file_path> [其他覆盖参数...]"
    exit 1
fi

# 检查配置文件是否存在
if [ ! -f "$CONFIG_FILE" ]; then
    echo "错误: 配置文件不存在于 '$CONFIG_FILE'"
    exit 1
fi

# --- 环境设置 ---

# 获取脚本所在的目录
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# 设置 PYTHONPATH，使其包含项目根目录 (RL_car2)
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"


# --- 从YAML生成命令行参数 ---

# 使用Python解析YAML并将其转换为 '--key value' 格式的字符串
# - 'y or {}' 确保在空文件时不会出错
# - 'if v is not None' 避免将 null 值转换为空参数
# - 'isinstance(v, bool)' 处理布尔值，转换为 'true'/'false' 字符串
# - 'isinstance(v, list)' 处理列表，将它们转换为空格分隔的字符串
CONFIG_ARGS=$(python3 -c "
import yaml
import sys

with open('$CONFIG_FILE', 'r') as f:
    data = yaml.safe_load(f)

args = []
for k, v in (data or {}).items():
    if v is not None:
        arg_name = f'--{k}'
        if isinstance(v, bool):
            # 将 Python的 True/False 转换为 shell 能理解的 true/false
            args.extend([arg_name, str(v).lower()])
        elif isinstance(v, list):
            # 将列表转换为多个值
            args.append(arg_name)
            args.extend(map(str, v))
        else:
            args.extend([arg_name, str(v)])

print(' '.join(args))
")

echo "======================================================================"
echo "PYTHONPATH 设置为: $PYTHONPATH"
echo "加载配置文件: $CONFIG_FILE"
echo "从配置文件生成的参数: ${CONFIG_ARGS}"
echo "命令行覆盖参数: ${OTHER_ARGS[@]}"
echo "======================================================================"

# --- 启动训练 ---

# 执行Python训练脚本
# 将从配置文件生成的参数放在前面，这样命令行覆盖参数(OTHER_ARGS)可以覆盖它们
# `set -- $CONFIG_ARGS` 将字符串解析为位置参数，以便处理带空格的值

COMMAND="python3 \"$SCRIPT_DIR/train_single.py\" ${CONFIG_ARGS} ${OTHER_ARGS[@]}"

echo "执行命令:"
echo "$COMMAND"
echo "----------------------------------------------------------------------"

# 使用eval来正确处理所有参数和引号
eval $COMMAND

echo "============================================="
echo "        ROSbot 强化学习训练脚本        "
echo "============================================="
echo "▶️  使用配置文件: $CONFIG_FILE"
echo "🐍 Python 脚本: $SCRIPT_DIR/train_single.py"
echo "============================================="

# 执行训练脚本，并传递配置文件路径
# Python脚本将负责解析此文件
python3 "$SCRIPT_DIR/train_single.py" --config "$CONFIG_FILE" "$@"

EXIT_CODE=$?

# --- 结果 --- 

if [ $EXIT_CODE -eq 0 ]; then
  echo "✅ 训练过程成功完成。"
else
  echo "⚠️ 训练过程因错误退出 (退出码: $EXIT_CODE)。"
fi

exit $EXIT_CODE
