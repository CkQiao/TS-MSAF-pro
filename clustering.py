import os
import random
import shutil

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def dbscan(Data, metric_list, Eps, MinPts):
    num = len(metric_list)  # 点的个数
    # print("点的个数："+str(num))
    unvisited = [i for i in range(num)]  # 没有访问到的点的列表
    # print(unvisited)
    visited = []  # 已经访问的点的列表
    C = {item: -1 for item in metric_list}
    # C为输出结果，默认是一个长度为num的值全为-1的列表
    # 用k来标记不同的簇，k = -1表示噪声点
    k = -1

    # 如果还有没访问的点
    while len(unvisited) > 0:
        # 随机选择一个unvisited对象
        p = random.choice(unvisited)
        unvisited.remove(p)
        visited.append(p)
        # N为p的epsilon邻域中的对象的集合
        N = []
        for i in range(num):
            if Data.loc[metric_list[i], metric_list[p]] <= Eps:  # and (i!=p):
                N.append(i)
        # 如果p的epsilon邻域中的对象数大于指定阈值，说明p是一个核心对象
        if len(N) >= MinPts:
            k = k + 1
            # print(k)
            C[metric_list[p]] = k
            # 对于p的epsilon邻域中的每个对象pi
            for pi in N:
                if pi in unvisited:
                    unvisited.remove(pi)
                    visited.append(pi)
                    # 找到pi的邻域中的核心对象，将这些对象放入N中
                    # M是位于pi的邻域中的点的列表
                    M = []
                    for j in range(num):
                        if Data.loc[metric_list[j], metric_list[pi]] <= Eps:  # and (j!=pi):
                            M.append(j)
                    if len(M) >= MinPts:
                        for t in M:
                            if t not in N:
                                N.append(t)
                # 若pi不属于任何簇，C[pi] == -1说明C中第pi个值没有改动
                if C[metric_list[pi]] == -1:
                    C[metric_list[pi]] = k
        # 如果p的epsilon邻域中的对象数小于指定阈值，说明p是一个噪声点
        else:
            C[metric_list[p]] = -1
    return C


def find_quartiles(df):
    # 展平矩阵为一维数组
    flattened_array = df.stack().values
    # 显示图形
    # 计算%3分位点
    first_quartile = np.quantile(flattened_array, 0.05)
    print(first_quartile)
    # 返回四分位数
    return first_quartile


metric_distance = pd.read_csv('./metric_distance.csv')
metric_distance = metric_distance.set_index(metric_distance.columns[0])
C = dbscan(metric_distance, metric_distance.index.tolist(), find_quartiles(metric_distance),
           10)
df = pd.DataFrame.from_dict(C, orient='index', columns=['Value'])

# 输出到 CSV 文件
df.to_csv('output.csv')
print('文件已输出')
source_folder = './selected'
output_folder = './pictures'
shutil.rmtree(output_folder,ignore_errors=True)


def move_files_based_on_dict(file_dict, source_folder, output_folder):
    # 定义不被Windows系统路径识别的字符列表
    bad_chars = ['<', '>', ':', '"', '/', '\\', '|', '?', '*']

    # 遍历字典
    for original_filename_prefix, category in file_dict.items():
        # 替换文件名前缀中的非法字符
        safe_filename_prefix = original_filename_prefix
        for char in bad_chars:
            safe_filename_prefix = safe_filename_prefix.replace(char, '_')

            # 拼接完整的文件名（替换后的前缀 + .png）
        safe_filename = f"{safe_filename_prefix}.png"
        # 构建目标文件夹路径
        target_folder = os.path.join(output_folder, str(category))
        # 构建源文件路径（使用替换后的文件名）
        source_file = os.path.join(source_folder, safe_filename)

        # 如果目标文件夹不存在，则创建它
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)

            # 检查源文件是否存在
        if os.path.exists(source_file):
            # 将文件移动到目标文件夹中
            shutil.copy(source_file, target_folder)
            print(f"Copied {safe_filename} to {target_folder}")
        else:
            print(f"File {safe_filename} not found in {source_folder}")

    print("All files have been copied to their respective folders.")


move_files_based_on_dict(C, source_folder, output_folder)
