import numpy as np
import os
import pandas as pd
import random
from progress.bar import Bar
from scipy.spatial.distance import cdist

INPUT_DIR = 'E:/project/TimeList/data4/original'
OUTPUT_DIR = 'E:/project/TimeList/data4/step1.5'

from scipy.spatial.distance import correlation
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from bayesian_changepoint_detection.bayesian_models import online_changepoint_detection
from bayesian_changepoint_detection.online_likelihoods import StudentT
from bayesian_changepoint_detection.hazard_functions import constant_hazard
from functools import partial
import bayesian_changepoint_detection.online_likelihoods as online_ll

def euclidean_multidim(x, y):
    return np.linalg.norm([x, y])


def dist(x, y, methods):
    match methods:
        case 1:
            result, _ = fastdtw(x, y, 1)
        case 2:
            result, _ = fastdtw(x, y, 2)
        case 3:
            result = euclidean(x, y)
        case 4:
            result = correlation(x, y)
        case _:
            result = None
    return result


def pretask(file_path, method=2):
    data = pd.read_csv(file_path)
    selected = data.columns
    np.seterr(divide='ignore', invalid='ignore')
    start_index = data['t'].iloc[0] + 11

    data.set_index('t', inplace=True)
    data = data.diff()
    data.drop(data.head(11).index, inplace=True)  # 从头去掉n行
    data.drop(data.tail(10).index, inplace=True)  # 从尾部去掉n行
    # 去掉重复索引的行
    bar = Bar(file_path, max=len(selected) - 1)
    data = data.loc[~data.index.duplicated(keep='first')]
    slide_dtw = pd.DataFrame()
    columns = selected
    for n in range(1, len(columns)):
        metric = columns[n]
        m = data[metric]
        record_dtw = []
        for i in range(0, len(m) - 150, 50):
            s1 = np.array(m[i:i + 100]).tolist()
            s2 = np.array(m[i + 50: i + 150]).tolist()
            distance = dist(s1, s2, method)
            record_dtw.append(distance)
        slide_dtw[metric] = record_dtw
        bar.next()
    bar.finish()

    simp_moving_avg = slide_dtw.rolling(window=10, min_periods=1).mean()
    return slide_dtw, simp_moving_avg, start_index


def mapping(data, area=100):  # 这里的data暂时不限定维数,DataFrame
    col = data.columns
    length = data.apply(lambda x: x.count(), axis=0).max()
    df = pd.DataFrame()
    for i in col:
        df[i] = np.interp(data[i], (data[i].min(), data[i].max()), (0, area))
    return df

def uniform_sample_and_calculate_distance(data_points, sample_size=100):
    # 如果数据点少于或等于sample_size，则直接使用所有数据点
    length = min(len(data_points), 100)
    if len(data_points) <= sample_size:
        sampled_points = data_points
    else:
        # 计算取样的间隔
        interval = len(data_points) // sample_size
        # 使用等间隔取样
        indices = np.arange(0, len(data_points), interval)
        sampled_points = data_points[indices]

        # 将数据点转换为二维坐标形式 (x, y)，其中x是顺序排列的
    x_values = np.arange(len(sampled_points))
    y_values = sampled_points
    points = list(zip(x_values, y_values))

    # 计算相邻点之间的欧氏距离
    distances = []
    for i in range(len(points) - 1):
        p1, p2 = points[i], points[i + 1]
        x1, y1 = p1
        x2, y2 = p2
        distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        distances.append(distance)

        # 计算平均欧氏距离
    average_distance = np.mean(distances) / 100 * length

    return average_distance


def calculate_distance(interval_vector):
    A = np.array(interval_vector)
    B = np.array(interval_vector)
    dist = cdist(A, B, metric='euclidean')
    return dist


def get_neighborhood(Data, metric_list, p, Eps, cluster):
    """计算点p的epsilon邻域，排除已经在cluster中的点"""
    M = [i for i in range(len(metric_list)) if metric_list[i] not in cluster]
    if len(M) == 0:
        return M
    N = [i for i in range(len(M)) if (
            Data.loc[metric_list[M[i]], p] <= Eps and
            i != p
    )]
    return N


def expand_cluster(Data, metric_list, Eps, MinPts, p, visited, cluster):
    """扩展簇"""
    visited.add(p)
    cluster.add(p)
    N = get_neighborhood(Data, metric_list, p, Eps, cluster)
    if len(N) == 0:
        return  # 如果点的邻域内点数少于MinPts，则不是核心点，返回
    for pi in N:
        if pi not in visited:
            expand_cluster(Data, metric_list, Eps, MinPts, pi, visited, cluster)


def dbscan(Data, metric_list, Eps, MinPts):
    num = len(metric_list)  # 点的个数
    visited = set()  # 已访问的点的集合
    cluster = set()  # 存储簇的集合
    C = [-1 for _ in range(num)]
    # 随机选择一个未访问的点开始搜索簇
    while metric_list and not visited:
        p = random.choice(metric_list)
        metric_list.remove(p)  # 从列表中移除已选择的点，避免重复选择
        if p not in visited:
            N = get_neighborhood(Data, metric_list, p, Eps, cluster)
            if len(N) >= MinPts:
                expand_cluster(Data, metric_list, Eps, MinPts, p, visited, cluster)
                break  # 找到一个簇后退出循环
    for i in cluster:
        C[i] = 0
    # 返回簇和已访问的点集合（可用于后续处理或调试）
    return C


def dbscan_area(data, eps, max_off, length):
    list_result = [1 for _ in range(length)]
    a = (eps + 1) * max_off * 100 / float(length)
    latest = []
    for i in range(eps, length, int(eps * 0.7) - 1):
        values = data[i - eps:i + eps - 1]
        distance = pd.DataFrame(calculate_distance(values))
        c = dbscan(distance, distance.index.tolist(), a, eps + 1)
        if i == eps:
            latest = c[-eps:]
            continue
        new = c[:eps]

        for j in range(eps):
            if new[j] == -1 and latest[j] == -1:
                list_result[(j + i - eps)] = -1
        latest = c[-eps:]
    dbscan_result = [idx for idx, value in enumerate(list_result) if value == -1]
    length = len(data[0])
    dbscan_result.append(length)
    return dbscan_result

def find_intervals(lst):
    lst.sort()
    intervals = []
    istart = lst[0]
    iend = lst[0]
    for i in range(1, len(lst)):
        if lst[i] - iend <= 10:
            iend = lst[i]
        else:
            if istart != iend:
                intervals.append([istart, iend])
            istart = lst[i]
            iend = lst[i]
    if istart != iend:
        intervals.append([istart, iend])
    return intervals

hazard_function = partial(constant_hazard, 250)
def find_interval(data):
    R, maxes = online_changepoint_detection(
        data, hazard_function, online_ll.StudentT(alpha=0.1, beta=.01, kappa=1, mu=0)
    )
    Nw = 10
    changepoints = [i for i, value in enumerate(maxes) if value <10]
    return changepoints

def onefile(filepath):
    DTW_result, swavg_result, start_index = pretask(filepath)
    metric_use0 = DTW_result
    metric_use1 = metric_use0.reset_index().rename(columns={'index': 't'})
    selected = metric_use1.columns

    ## 聚类变点检测
    # metric_use1 = metric_use1.astype(float)
    # metric_use = mapping(metric_use1)
    # filtered_result = {}
    # for cu in selected:
    #     if cu == 't':
    #         continue
    #     distance = uniform_sample_and_calculate_distance(metric_use[cu])
    #     temp_metric = metric_use[['t', cu]].to_numpy()
    #     length_metric = len(temp_metric)
    #     db_result = dbscan_area(temp_metric, min(10, int(length_metric / 10)), distance, length_metric)
    #     filtered_result[cu] = pd.Series(find_intervals(db_result))
    # result = pd.DataFrame(filtered_result)

    # ## 贝叶斯变点检测版本
    filtered_result = {}
    for column in selected:
        db_result = find_interval(metric_use1[column])
        filtered_result[column] = pd.Series(find_intervals(db_result))
    result = pd.DataFrame(filtered_result)

    rel_path = os.path.relpath(filepath, INPUT_DIR)
    if not os.path.exists(os.path.split(os.path.join(OUTPUT_DIR, rel_path))[0]):
        os.makedirs(os.path.split(os.path.join(OUTPUT_DIR, rel_path))[0])
    result.to_csv(os.path.join(OUTPUT_DIR, rel_path))

for root, dirs, files in os.walk(INPUT_DIR, topdown=False):
    for file in files:
        if file == 'ground_truth.csv':
            continue
        file_path = str(os.path.join(root, file))
        onefile(file_path)
