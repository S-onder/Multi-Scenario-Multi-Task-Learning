import re
import sys
import pandas as pd

# 定义正则表达式提取信息，仅匹配 val 相关内容
pattern_auc = (
    r"Epoch (\d+) val loss is ([\d.]+), (total scene|scene \d+), "
    r"click auc is ([\d.]+), like auc is ([\d.]+), "
    r"comment auc is ([\d.]+), 5s auc is ([\d.]+), 10s auc is ([\d.]+), 18s auc is ([\d.]+)"
)

pattern_ndcg = (
    r"Epoch (\d+) val loss is ([\d.]+), (total scene|scene \d+), "
    r"click ndcg is ([\d.]+), like ndcg is ([\d.]+), "
    r"comment ndcg is ([\d.]+), 5s ndcg is ([\d.]+), 10s ndcg is ([\d.]+), 18s ndcg is ([\d.]+)"
)


if __name__ == "__main__":
    model_name = sys.argv[1]
    try:
        # 从标准输入逐行读取日志
        print('epoch', '\t', 'model', '\t', 'scene', '\t', 'val_loss', '\t', 'ratio_type', '\t', 'click_ratio', '\t', 'like_ratio', '\t', 'comment_auc', '\t', '5s_ratio', '\t', '10s_ratio', '\t', '18s_ratio')
        for line in sys.stdin:
            match_auc = re.search(pattern_auc, line)
            match_ndcg = re.search(pattern_ndcg, line)
            if match_auc:
                # 提取匹配数据
                epoch = int(match_auc.group(1))
                scene = match_auc.group(3)
                val_loss = float(match_auc.group(2))
                click_auc = float(match_auc.group(4))
                like_auc = float(match_auc.group(5))
                comment_auc = float(match_auc.group(6))
                five_s_auc = float(match_auc.group(7))
                ten_s_auc = float(match_auc.group(8))
                eighteen_s_auc = float(match_auc.group(9))
                # 格式化输出
                print(f"{epoch}\t{model_name}\t{scene}\t{val_loss:.6f}\t{'auc'}\t{click_auc:.6f}\t{like_auc:.6f}\t{comment_auc:.6f}\t{five_s_auc:.6f}\t{ten_s_auc:.6f}\t{eighteen_s_auc:.6f}")
            if match_ndcg:
                            # 提取匹配数据
                epoch = int(match_ndcg.group(1))
                scene = match_ndcg.group(3)
                val_loss = float(match_ndcg.group(2))
                click_ndcg = float(match_ndcg.group(4))
                like_ndcg = float(match_ndcg.group(5))
                comment_ndcg = float(match_ndcg.group(6))
                five_s_ndcg = float(match_ndcg.group(7))
                ten_s_ndcg = float(match_ndcg.group(8))
                eighteen_s_ndcg = float(match_ndcg.group(9))
                # 格式化输出
                print(f"{epoch}\t{model_name}\t{scene}\t{val_loss:.6f}\t{'ndcg'}\t{click_ndcg:.6f}\t{like_ndcg:.6f}\t{comment_ndcg:.6f}\t{five_s_ndcg:.6f}\t{ten_s_ndcg:.6f}\t{eighteen_s_ndcg:.6f}")
    except KeyboardInterrupt:
        # 支持 Ctrl+C 提前终止
        pass
