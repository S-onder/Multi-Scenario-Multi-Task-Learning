import re
import sys
import pandas as pd

# 定义正则表达式提取信息，仅匹配 val 相关内容
pattern = (
    r"Epoch (\d+) val loss is ([\d.]+), (total scene|scene \d+), "
    r"click auc is ([\d.]+), like auc is ([\d.]+), "
    r"follow auc is ([\d.]+), 5s auc is ([\d.]+), 10s auc is ([\d.]+), 18s auc is ([\d.]+)"
)

# 存储结果的列表
extracted_data = []

try:
    # 从标准输入逐行读取日志
    print('epoch', '\t', 'scene', '\t', 'val_loss', '\t', 'click_auc', '\t', 'like_auc', '\t', 'follow_auc', '\t', '5s_auc', '\t', '10s_auc', '\t', '18s_auc')
    for line in sys.stdin:
        match = re.search(pattern, line)
        if match:
            # 提取匹配数据
            epoch = int(match.group(1))
            scene = match.group(3)
            val_loss = float(match.group(2))
            click_auc = float(match.group(4))
            like_auc = float(match.group(5))
            follow_auc = float(match.group(6))
            five_s_auc = float(match.group(7))
            ten_s_auc = float(match.group(8))
            eighteen_s_auc = float(match.group(9))

            # 存储提取到的数据
            values = [epoch, scene, val_loss, click_auc, like_auc, follow_auc, five_s_auc, ten_s_auc, eighteen_s_auc]
            extracted_data.append(values)

            # 格式化输出
            print(f"{epoch}\t{scene}\t{val_loss:.4f}\t{click_auc:.4f}\t{like_auc:.4f}\t{follow_auc:.4f}\t{five_s_auc:.4f}\t{ten_s_auc:.4f}\t{eighteen_s_auc:.4f}")

    # 将提取到的数据存储到 DataFrame 中
    if extracted_data:
        columns = ['epoch', 'scene', 'val_loss', 'click_auc', 'like_auc', 'follow_auc', '5s_auc', '10s_auc', '18s_auc']
        df = pd.DataFrame(extracted_data, columns=columns)
        # 可以在这里对 DataFrame 进行进一步的数据分析
       # print(df)

except KeyboardInterrupt:
    # 支持 Ctrl+C 提前终止
    pass
