import glob
import os
import csv
import random
from pathlib import Path
from scipy.spatial.distance import squareform, pdist
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN

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

def dbscan(Data, metric_list, Eps, MinPts):
    num = len(metric_list)  # 点的个数
    # print("点的个数："+str(num))
    unvisited = [i for i in range(num)]  # 没有访问到的点的列表
    # print(unvisited)
    visited = []  # 已经访问的点的列表
    C = [-1 for i in range(num)]
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
            C[p] = k
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
                if C[pi] == -1:
                    C[pi] = k
        # 如果p的epsilon邻域中的对象数小于指定阈值，说明p是一个噪声点
        else:
            C[p] = -1
    return C


def merge_intervals(intervals):
    # 如果没有区间，返回空列表
    if not intervals:
        return []

        # 首先按照区间的起始位置进行排序
    intervals.sort(key=lambda x: x[0])

    # 初始化结果列表
    merged = []

    # 当前正在合并的区间
    current_interval = intervals[0]

    # 遍历排序后的区间列表
    for interval in intervals[1:]:
        # 如果当前区间的结束位置大于等于下一个区间的起始位置，说明它们有重叠
        if current_interval[1] >= interval[0]:
            # 更新当前区间的结束位置为两个区间中较大的结束位置
            current_interval = (current_interval[0], max(current_interval[1], interval[1]))
        else:
            # 如果没有重叠，将当前区间添加到结果列表中，并开始合并下一个区间
            merged.append(current_interval)
            current_interval = interval

            # 不要忘记将最后一个区间添加到结果列表中
    merged.append(current_interval)

    return merged

def superposition(data_dict: dict, filename: str) -> list:
    point_list = []
    for i, start_ends in enumerate(data_dict.values()):
        for point in start_ends:
            point_list.append(point)
    points_array = np.array(point_list)
    distances = pdist(points_array, 'euclidean')
    distance_matrix = squareform(distances)
    colume_points = [f'point{i}' for i in range(len(point_list))]
    result_dbscan = dbscan(pd.DataFrame(distance_matrix,index=colume_points,columns=colume_points), colume_points,2,5)
    contiguous_intervals = list(set([point_list[i] for i in range(len(result_dbscan)) if not result_dbscan[i] == -1]))
    interval_list = []
    # 将连续区间转换为字典列表形式，包含文件名、开始时间和结束时间
    contiguous_intervals = merge_intervals(contiguous_intervals)
    interval_list.extend([{'trace_name': filename, 'root_cause_start': start, 'root_cause_end': end} for start, end in contiguous_intervals])

    return interval_list


def write_data_to_csv(data: list, csv_filename: str):
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
            data_dict['root_cause_start'] = int(data_dict['root_cause_start']) * 50 + 11
            data_dict['root_cause_end'] = int(data_dict['root_cause_end']) * 50 + 11
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

def action(input_dir, outputfile):
    if os.path.exists(outputfile):
        os.remove(outputfile)
    for root, dirs, files in os.walk(input_dir, topdown=False):
        for file in files:
            if file == 'ground_truth.csv':
                continue
            file_path = str(os.path.join(root, file))
            dict1, length = dataframe_to_indexed_values_dict(file_path)
            file = Path(file_path).stem
            dict2 = superposition(dict1, file)
            write_data_to_csv(dict2, outputfile)


if __name__ == '__main__':


    action(input_dir='E:/project/TimeList/data4/step1.5/train',
           outputfile='E:/project/TimeList/data4/step2.5/train.csv')
    action(input_dir='E:/project/TimeList/data4/step1.5/test',
           outputfile='E:/project/TimeList/data4/step2.5/test.csv')
