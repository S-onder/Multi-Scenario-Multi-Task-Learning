#!/bin/bash
# 检查是否提供了参数
if [ $# -ne 2 ]; then
    echo "Usage: $0 <config_file> <log_file>"
    exit 1
fi
config_file="$1"
log_file="$2"
echo "Running with config file: $config_file"
echo "Logging to: log/${log_file}.txt"
nohup python -u main.py --config $1 >"log/${log_file}.log" 2>&1 &
echo "Staring Trainging Model"

