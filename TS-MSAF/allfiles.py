import glob
import os
import csv
import random
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# selected = ['1_executor_cpuTime_count', '1_executor_shuffleLocalBytesRead_count',
#             '1_executor_shuffleRecordsRead_count', '1_executor_shuffleRemoteBytesRead_count',
#             '1_executor_shuffleTotalBytesRead_count', '1_jvm_heap_committed_value', '1_jvm_heap_usage_value',
#             '1_jvm_heap_used_value', '1_jvm_pools_PS-Eden-Space_max_value', '1_jvm_pools_PS-Eden-Space_usage_value',
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
selected = [f'metric{i + 1}' for i in range(38)]


def find_contiguous_intervals(arr, threshold):
    """找到数组中大于阈值的连续区间"""
    intervals = []
    start = None
    for i, val in enumerate(arr):
        if val > threshold and start is None:
            start = i
        elif val <= threshold and start is not None:
            intervals.append((start, i))
            start = None
    if start is not None:
        intervals.append((start, len(arr)))
    return intervals


def superposition(data_dict: dict, weight_dict: dict, length: int, threshold: float, filename: str) -> list:
    # 初始化一个长度为length的零数组，用于存储叠加结果
    result = np.zeros(length)

    # 遍历data_dict中的每一个指标（即每一个键）
    for i, (column, intervals) in enumerate(data_dict.items()):
        # 获取当前指标的权重
        w = weight_dict.get(column, 0)
        if w == 0:
            # 如果权重为0，则跳过当前指标
            continue
        # color = colors[i % len(colors)]
        # 遍历小区间数据
        for interval in intervals:
            start, end = interval
            result[start:end] += w
            # 绘制累计折线图
    # 找到大于门限的连续区间
    contiguous_intervals = find_contiguous_intervals(result, threshold)
    interval_list = []
    # 将连续区间转换为字典列表形式，包含文件名、开始时间和结束时间
    interval_list.extend([{'trace_name': filename, 'root_cause_start': start, 'root_cause_end': end} for start, end in
                          contiguous_intervals])

    return interval_list


def write_data_to_csv(data: list, start_index, csv_filename: str):
    # 检查文件是否存在
    if os.path.exists(csv_filename):
        # 读取文件的前几行来检查格式
        with open(csv_filename, mode='r', newline='') as csv_file:
            reader = csv.DictReader(csv_file)
            # 假设如果列标题正确，则格式也正确
            header = next(reader, None)
            if header and set(header.keys()) == {'trace_name', 'root_cause_start', 'root_cause_end'}:
                # 格式正确，以追加模式打开文件
                mode = 'a'
            else:
                # 格式不正确，覆盖文件
                mode = 'w'
    else:
        # 文件不存在，创建新文件
        mode = 'w'

        # 使用相应的模式打开文件
    with open(csv_filename, mode=mode, newline='') as csv_file:
        if mode == 'w':
            # 创建一个csv writer对象并写入列头
            csv_writer = csv.DictWriter(csv_file, fieldnames=['trace_name', 'root_cause_start', 'root_cause_end'])
            csv_writer.writeheader()
        else:
            # 追加模式时不需要写入列头
            csv_writer = csv.DictWriter(csv_file, fieldnames=['trace_name', 'root_cause_start', 'root_cause_end'])

            # 写入数据
        for data_dict in data:
            # 修改start和end项的值
            data_dict['root_cause_start'] = int(data_dict['root_cause_start']) * 50 + 1
            data_dict['root_cause_end'] = int(data_dict['root_cause_end']) * 50 + 1
            # 写入修改后的行
            csv_writer.writerow(data_dict)


def dataframe_to_indexed_values_dict(filepath):
    df = pd.read_csv(filepath, usecols=selected)
    # 创建一个空字典来存储值
    dict_of_indexed_values = {}
    length = 0
    # 遍历DataFrame的列

    for column in df.columns:
        # 初始化一个空列表来存储区间
        intervals = []
        # 遍历该列的每个值
        for value in df[column]:
            # 检查值是否是字符串类型
            if isinstance(value, str):
                # 去除字符串两端的空格和方括号，然后分割成start和end
                start, end = map(int, value.strip('[]').split(','))
                intervals.append((start, end))
                # 将列索引和区间列表作为键值对添加到字典中
        dict_of_indexed_values[column] = intervals
    for intervals in dict_of_indexed_values.values():
        # 遍历每个键的区间列表
        for interval in intervals:
            start, end = interval
            # 更新最大end值
            length = max(length, end + 1)
    return dict_of_indexed_values, length


def get_starttime():
    start_index = {}
    csv_file = './timelist.csv'
    with open(csv_file, 'r', newline='') as csvfile:
        # 创建一个CSV阅读器对象
        csvreader = csv.reader(csvfile)
        # 遍历CSV文件的每一行
        for row in csvreader:
            # 确保行中有两个元素
            if len(row) == 2:
                key = row[0]
                value = row[1]
                # 将键值对添加到字典中
                start_index[key] = int(value)
            else:
                # 如果行不是预期的格式，可以打印警告或跳过该行
                print(f"Skipping row with unexpected number of columns: {row}")
    return start_index


def action(weight, threshold, input_dir, outputfile):
    if os.path.exists(outputfile):
        os.remove(outputfile)
    start_index = [1]
    for root, dirs, files in os.walk(input_dir, topdown=False):
        for file in files:
            if file == 'ground_truth.csv':
                continue
            file_path = str(os.path.join(root, file))
            dict1, length = dataframe_to_indexed_values_dict(file_path)
            file = Path(file_path).stem
            dict2 = superposition(dict1, weight, length, threshold, file)
            write_data_to_csv(dict2, 1, outputfile)


if __name__ == '__main__':
    weight = {}
    for cu in selected:
        weight[cu] = 1
    threshold = 10

    action(weight, threshold, input_dir='E:/project/TimeList/data2/task/step1/test/machine',
           outputfile='E:/project/TimeList/data2/task/step1/test/machine/output.csv')

