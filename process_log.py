import re
import sys
import pandas as pd

# 定义正则表达式提取信息
pattern = (
    r"Epoch (\d+) val loss is ([\d.]+), click auc is ([\d.]+), like auc is ([\d.]+), "
    r"follow auc is ([\d.]+), 5s auc is ([\d.]+), 10s auc is ([\d.]+), 18s auc is ([\d.]+)"
)

# 存储结果的列表
# results = []

try:
    # 从标准输入逐行读取日志
    print('epoch', '\t', 'val_loss', '\t', 'click_auc', '\t', 'like_auc', '\t', 'follow_auc', '\t', '5s_auc', '\t', '10s_auc', '\t', '18s_auc')
    for line in sys.stdin:
        match = re.search(pattern, line)
        if match:
            # 提取匹配数据并转为 float
            results = list(map(float, match.groups()))
            print(str(results[0]), '\t', results[1], '\t', results[2], '\t', results[3], '\t', results[4], '\t', results[5], '\t', results[6], '\t', results[7])
except KeyboardInterrupt:
    # 支持 Ctrl+C 提前终止
    pass
