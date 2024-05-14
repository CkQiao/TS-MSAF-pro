from random import random

import pandas as pd
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from progress.bar import Bar
from sklearn.cluster import DBSCAN

# 假设你的CSV文件列表如下
csv_files = ['bursty_input.csv', 'bursty_input_crash.csv', 'cpu_contention.csv', 'normal.csv', 'process_failure.csv',
             'stalled_input.csv']

# 读取所有CSV文件并合并它们的数据
all_data = []
for file in csv_files:
    df = pd.read_csv(file)


    def convert_to_float_list(s):
        try:
            return [float(x.strip()) for x in s.strip('[]').split(',')]
        except (ValueError, AttributeError):
            return []


    df['标准差'] = df['标准差'].apply(convert_to_float_list)
    df['均值'] = df['均值'].apply(convert_to_float_list)
    df.columns = ['指标名', '标准差', '均值']  # 确保所有文件的列名都是统一的
    df['异常名'] = file  # 添加一个列来标识数据来源于哪个文件
    all_data.append(df)
# 合并所有DataFrame
combined_data = pd.concat(all_data, ignore_index=True)


def linear_map(valuess, min_orig, max_orig, min_new, max_new):
    new_valuess = []
    for values in valuess:
        # 检查值是否在原始范围内，避免除以零错误
        new_values = []
        for value in values:
            if min_orig == max_orig:
                new_value = min_new  # 如果原始范围是一个点，则所有值都映射到目标范围的最小值
            else:
                # 线性映射公式
                new_value = min_new + ((value - min_orig) / (max_orig - min_orig)) * (max_new - min_new)
            new_values.append(new_value)
        new_valuess.append(new_values)
    return new_valuess


# 对每个独特的“指标名”进行归一化处理
scaler_dict = {}
normalized_data = []
for group_name, group in combined_data.groupby('指标名'):
    # 获取该组的所有值列表
    std_list = group['标准差'].tolist()
    flat_list = [item for sublist in std_list for item in sublist]
    max_value = max(flat_list)
    std_list = linear_map(std_list, 0, max_value, 0, 1)

    mean_list = group['均值'].tolist()
    flat_list = [item for sublist in mean_list for item in sublist]
    max_value = max(flat_list)
    min_value = min(flat_list)
    mean_list = linear_map(mean_list, min_value, max_value, 0, 1)
    if len(mean_list) < 2:
        continue
    # 将归一化后的值和文件名添加到新列表
    for stds, means, filename in zip(std_list, mean_list, group['异常名']):
        normalized_data.append({
            '指标名': group_name,
            '标准差': stds,
            '均值': means,
            '异常名': filename
        })

    # 将归一化后的数据转换为DataFrame
normalized_df = pd.DataFrame(normalized_data)
normalized_df.to_csv('result.csv', index=False)
print('文件已输出')
# 绘制散点图
# selected = ['1_executor_cpuTime_count', '1_executor_shuffleLocalBytesRead_count',
#             '1_executor_shuffleRecordsRead_count',
#             '1_executor_shuffleRemoteBytesRead_count', '1_executor_shuffleTotalBytesRead_count',
#             '1_jvm_heap_committed_value', '1_jvm_heap_usage_value', '1_jvm_heap_used_value',
#             '1_jvm_pools_PS-Eden-Space_max_value', '1_jvm_pools_PS-Eden-Space_usage_value',
#             '1_jvm_pools_PS-Eden-Space_used_value', '1_jvm_pools_PS-Old-Gen_used_value',
#             '1_jvm_pools_PS-Survivor-Space_committed_value', '1_jvm_pools_PS-Survivor-Space_max_value',
#             '1_jvm_pools_PS-Survivor-Space_usage_value', '1_jvm_pools_PS-Survivor-Space_used_value',
#             '1_jvm_total_committed_value', '1_jvm_total_used_value', '2_jvm_heap_usage_value', '2_jvm_heap_used_value',
#             '2_jvm_pools_PS-Eden-Space_usage_value', '2_jvm_pools_PS-Eden-Space_used_value',
#             '2_jvm_pools_PS-Survivor-Space_committed_value', '2_jvm_pools_PS-Survivor-Space_max_value',
#             '2_jvm_pools_PS-Survivor-Space_used_value', '2_jvm_total_used_value',
#             'driver_BlockManager_memory_memUsed_MB_value', 'driver_BlockManager_memory_onHeapMemUsed_MB_value',
#             'driver_BlockManager_memory_remainingMem_MB_value',
#             'driver_BlockManager_memory_remainingOnHeapMem_MB_value']
#
# for metric in normalized_df['指标名'].unique():
#     # 筛选出当前指标名的数据
#     metric_df = normalized_df[normalized_df['指标名'] == metric]
#
#     # 准备绘图
#     plt.figure(figsize=(10, 6))
#
#     # 遍历每个异常名及其对应的值
#     for idx, row in metric_df.iterrows():
#         plt.scatter(row['标准差'], row['均值'], label=row['异常名'])
#
#         # 添加图例
#     plt.legend()
#
#     # 添加标题和轴标签
#     plt.title(f'{metric} ')
#     plt.xlabel('std')
#     plt.ylabel('mean')
#
#
#     def sanitize_filename(filename):
#         # 定义Windows不支持的字符列表
#         bad_chars = ['<', '>', ':', '"', '/', '\\', '|', '?', '*']
#         # 替换每个不支持的字符为下划线
#         for char in bad_chars:
#             filename = filename.replace(char, '_')
#         return filename
#
#         # 显示图形
#     plt.savefig(os.path.join('./pictures', f'{sanitize_filename(metric)}.png'))
#     plt.close()
df = normalized_df
# 3. 初始化一个空的字典来存储指标名和其对应的坐标点
points_by_indicator = {}
for metric in df['指标名'].unique():
    # 筛选出当前指标名的数据
    metric_df = df[df['指标名'] == metric]
    for idx, row in metric_df.iterrows():
        indicator = row['指标名']
        std_devs = row['标准差']
        means = row['均值']
        abnormality_codes = [csv_files.index(row['异常名'])] * len(std_devs)
        assert len(std_devs) == len(means), "标准差和均值的列表长度不相等"
        # 创建三维坐标点
        if indicator not in points_by_indicator:
            points_by_indicator[indicator] = []
        for i in range(len(std_devs)):
            points = list([std_devs[i], means[i], abnormality_codes[i]])
            points_by_indicator[indicator].extend([points])
sizes = {indicator: len(points) for indicator, points in points_by_indicator.items()}
# 6. 找出最常见的点集大小
most_common_size = Counter(sizes).most_common(1)[0][1]
# 7. 过滤出具有最常见大小的点集
filtered_points_by_indicator = {
    indicator: points for indicator, points in points_by_indicator.items()
    if len(points) == most_common_size
}
filtered_points_by_indicator = pd.DataFrame(filtered_points_by_indicator)
print(filtered_points_by_indicator.shape)


def min_euclidean_distance(point_set_a, point_set_b):
    assert point_set_a.shape == point_set_b.shape, "点集形状必须相同"

    min_distance = 0
    for i in range(point_set_a.shape[0]):
        distance = sum(abs(a - b) for a, b in zip(point_set_a.loc[i], point_set_b.loc[i]))
        min_distance += distance

    return min_distance


def compute_distance_matrix(df):
    # 获取DataFrame的列数
    n_cols = df.shape[1]
    bar = Bar('progress', maxval=n_cols)
    # 初始化距离矩阵
    distance_matrix = np.zeros((n_cols, n_cols))

    # 遍历每一对列，计算它们之间的最小欧式距离
    for i in range(n_cols):
        for j in range(i, n_cols):
            point_set_a = df.iloc[:, i]
            point_set_b = df.iloc[:, j]
            distance = min_euclidean_distance(point_set_a, point_set_b)
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance  # 因为距离矩阵是对称的，所以也设置上三角的值
        bar.next()
    return distance_matrix


metric_distance = compute_distance_matrix(filtered_points_by_indicator)
metric_distance = pd.DataFrame(metric_distance,index=filtered_points_by_indicator.columns,columns=filtered_points_by_indicator.columns)
metric_distance.to_csv('./metric_distance.csv')
