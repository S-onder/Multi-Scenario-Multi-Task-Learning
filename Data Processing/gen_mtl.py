import random
import csv
import os
import pandas as pd
from tqdm import tqdm

# 源文件路径
path = '/data/QK-article.csv'

# 分割后的子文件存储路径

workspace = '/data/mtl_task_article'

def split_large_csv(path, workspace, user_id_limit=1000017, rows_per_file=1000000):
    """
    分割大CSV文件到多个小文件。
    
    参数:
        path (str): 源CSV文件路径。
        workspace (str): 分割后的文件存储路径。
        user_id_limit (int): 处理用户ID小于该值的行。
        rows_per_file (int): 每个子文件包含的最大行数。
    """
    # 创建目标目录
    os.makedirs(workspace, exist_ok=True)
    
    # 初始化变量
    file_index = 0  # 子文件编号
    row_count = 0   # 当前子文件的行数
    csvwriter = None  # 当前文件的写入器
    current_file = None  # 当前文件对象
    
    with open(path, 'r', newline='') as file:
        csvreader = csv.reader(file)
        header = next(csvreader)  # 读取表头
        
        for row in csvreader:
            # 筛选 user_id
            user_id = int(row[0])
            if user_id >= user_id_limit:
                continue
            
            # 如果需要创建新文件
            if row_count == 0 or row_count >= rows_per_file:
                # 如果有打开的文件，先关闭
                if current_file:
                    current_file.close()
                
                # 更新文件索引和路径
                file_index += 1
                new_file_path = os.path.join(workspace, f'QK_article_part_{file_index}.csv')
                
                # 创建新文件并写入表头
                current_file = open(new_file_path, 'w', newline='')
                csvwriter = csv.writer(current_file)
                csvwriter.writerow(header)  # 写入表头
                row_count = 0  # 重置行计数
            
            # 写入数据
            csvwriter.writerow(row)
            row_count += 1
    
    # 最后关闭文件
    if current_file:
        current_file.close()
    
    print(f"分割完成，共生成 {file_index} 个子文件。")

if __name__ == '__main__':
    workspace = '/data/mtl_task_article'
    path = '/data/QK-article.csv'
    split_large_csv(path,workspace)

with open(path, 'r', newline='') as file:
    csvreader = csv.reader(file)
    a = next(csvreader)
    #print(a)
    i = j = 0
    for row in csvreader:
        # 每 10000 个就 j 加 1，然后就有一个新的文件名。
        if i % 1000000 == 0:
            j += 1
            print("完成第{}个100w数据".format(j-1))
#
        csv_path = os.path.join(workspace, 'QK_article1M.csv')
        user_id = int(row[0])
        if user_id < 1000017:
#         # 不存在此文件的时候，就创建
            if not os.path.exists(os.path.dirname(csv_path)):
                os.makedirs(os.path.dirname(csv_path))
                with open(csv_path, 'w', newline='') as file:
                    csvwriter = csv.writer(file)
                    csvwriter.writerow(row)
                i += 1
    #         # 存在的时候就往里面添加
            else:
                with open(csv_path, 'a', newline='') as file:
                    csvwriter = csv.writer(file)
                    csvwriter.writerow(row)
                i += 1

path = '/data/mtl_task_article/QK_article1M.csv'
source_data = pd.read_csv(path)
# source_data.columns = ['user_id', 'item_id', 'click', 'follow', 'like', 'share', 'short_v', 'play_times', 'gender', 'age']
pos_data = source_data[source_data.click.isin([1])]
user_history = pos_data.groupby('user_id').item_id.apply(list).to_dict()
del pos_data
user_hist = {}
user_target = {}
for key, value in user_history.items():
    if len(value) <= 10:
        if len(value) > 1:
            user_hist[key] = value[:-1] + (10 - len(value[:-1])) * [0]
            user_target[key] = value[-1:]
        else:
            user_hist[key] = [0] * 10
            user_target[key] = value
    else:
        user_hist[key] = value[:10]
        user_target[key] = value[10:]
del_list = []
for key, value in tqdm(user_hist.items()):
    for v in value:
        del_list.append([key, v])

def del_data(s_data, user_hist, i):
    print('++++++++{}+++++++'.format(i))
    new = []
    for user, value in tqdm(user_hist.items()):
        if user > i * 10000 and user <= (i+1) * 10000:
            data = s_data[s_data['user_id'] == user]
            tmp_data = data[~data.item_id.isin(value)].values.tolist()
            new.extend(tmp_data)
    return new

max_len = 1100000
times = max_len // 10000
new_list = []

for i in range(times):
    print("+++++++++times{}+++++++++".format(i))
    data = source_data[(source_data['user_id'] > i * 10000) & (source_data['user_id'] <= (i+1) * 10000)]
    new = del_data(data, user_hist, i)
    new_list.extend(new)
    # if len(new_list) == 100000 or i == times - 1:
new_data = pd.DataFrame(new_list, columns=source_data.columns)

hist_1, hist_2, hist_3, hist_4, hist_5, hist_6, hist_7, hist_8, hist_9, hist_10 = {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
for user, value in tqdm(user_hist.items()):
    hist_1[user] = value[0]
    hist_2[user] = value[1]
    hist_3[user] = value[2]
    hist_4[user] = value[3]
    hist_5[user] = value[4]
    hist_6[user] = value[5]
    hist_7[user] = value[6]
    hist_8[user] = value[7]
    hist_9[user] = value[8]
    hist_10[user] = value[9]

for i in range(1, 11):
    new_data['hist_{}'.format(i)] = new_data['user_id']
    # new_data['hist_{}'.format(i)] = new_data['hist_{}'.format(i)].map(hist)
new_data['hist_1'] = new_data['hist_1'].map(hist_1)
new_data['hist_2'] = new_data['hist_2'].map(hist_2)
new_data['hist_3'] = new_data['hist_3'].map(hist_3)
new_data['hist_4'] = new_data['hist_4'].map(hist_4)
new_data['hist_5'] = new_data['hist_5'].map(hist_5)
new_data['hist_6'] = new_data['hist_6'].map(hist_6)
new_data['hist_7'] = new_data['hist_7'].map(hist_7)
new_data['hist_8'] = new_data['hist_8'].map(hist_8)
new_data['hist_9'] = new_data['hist_9'].map(hist_9)
new_data['hist_10'] = new_data['hist_10'].map(hist_10)
new_data.to_csv('/data/mtl_task_article/mtl_data_1M.csv', header=True, index=False)

