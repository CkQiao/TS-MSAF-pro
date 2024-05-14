import pandas as pd


def fileread(real_file, output_file):
    real = pd.read_csv(real_file)
    predicted = pd.read_csv(output_file)
    real_intervals = {}
    predicted_intervals = {}
    for filename in set(real['trace_name']):
        real_intervals[filename] = []
        for index, row in real[real['trace_name'] == filename].iterrows():
            sub_interval_start = row['root_cause_start']
            sub_interval_end = row['root_cause_end']
            real_intervals[filename].append((sub_interval_start, sub_interval_end))
    for filename in set(predicted['trace_name']):
        predicted_intervals[filename] = []
        for index, row in predicted[predicted['trace_name'] == filename].iterrows():
            sub_interval_start = row['root_cause_start']
            sub_interval_end = row['root_cause_end']
            predicted_intervals[filename].append((sub_interval_start, sub_interval_end))
    return real_intervals, predicted_intervals
def calculate_overlap(interval_a, interval_b):
    # 辅助函数，计算两个区间的重叠长度
    start_overlap = max(interval_a[0], interval_b[0])
    end_overlap = min(interval_a[1], interval_b[1])
    return max(end_overlap - start_overlap, 0)

def merge_intervals(intervals):
    if not intervals:
        return []
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]
    for interval in intervals[1:]:
        if merged[-1][1] >= interval[0] - 1:  # 如果重叠，则合并
            merged[-1] = (merged[-1][0], max(merged[-1][1], interval[1]))
        else:
            merged.append(interval)
    return merged


def calculate_kpi(list_a, list_b):
    # 初始化指标
    true_positive_length = 0  # 真阳区间总长度
    false_negative_length = 0  # 假阴区间总长度
    false_positive_length = 0  # 假阳区间总长度
    true_negative_length = 0  # 真阴区间总长度（这里需要额外的计算）

    # 遍历listA的每个key_a和intervals_a
    for key_a, intervals_a in list_a.items():
        # 遍历listB的每个key_b，检查是否以key_a开头
        for key_b, intervals_b in list_b.items():
            if key_b.startswith(key_a):
                overlap_length = 0
                # 遍历intervals_a的每个interval_a
                for interval_a in intervals_a:
                    # 遍历intervals_b的每个interval_b
                    for interval_b in intervals_b:
                        overlap_length += calculate_overlap(interval_a, interval_b)
                if overlap_length > 0:
                    true_positive_length += overlap_length
                else:
                    false_negative_length += interval_a[1] - interval_a[0]
                    # 计算listA中当前key_a的真阴区间长度

        # 累加total_length_a和total_overlap_a
        total_length_a = sum(interval[1] - interval[0] for interval in intervals_a)
        total_overlap_a = sum(calculate_overlap(interval_a, interval_b)
                              for interval_a in intervals_a
                              for key_b, intervals_b in list_b.items()
                              if key_b.startswith(key_a)
                              for interval_b in intervals_b)

        true_negative_length += max(total_length_a - total_overlap_a, 0)

        # 遍历listB中不在listA的键
    for key_b in list_b:
        if not any(key_b.startswith(key_a) for key_a in list_a):
            intervals_b = list_b[key_b]

            false_positive_length += sum(interval[1] - interval[0] for interval in intervals_b)
            # 计算指标
    total_positive_length = true_positive_length + false_negative_length
    total_negative_length = true_negative_length + false_positive_length
    total_length = total_positive_length + total_negative_length

    precision = true_positive_length / (
            true_positive_length + false_positive_length) if total_positive_length > 0 else 0
    recall = true_positive_length / total_positive_length if total_positive_length > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    accuracy = (true_positive_length + true_negative_length) / total_length if total_length > 0 else 0

    return {
        'true_positive_length': true_positive_length,
        'false_negative_length': false_negative_length,
        'false_positive_length': false_positive_length,
        'true_negative_length': true_negative_length,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'accuracy': accuracy
    }


def KPIs(real_file, output_file):
    truth, predict = fileread(real_file, output_file)
    kpi_results = calculate_kpi(truth, predict)
    return kpi_results


if __name__ == '__main__':
    kpi_results1 = KPIs('E:/project/TimeList/data4/step2.5/train.csv',
                        'E:/project/TimeList/data4/label/train.csv')
    print('train')
    for key, value in kpi_results1.items():
        print(f"{key}: {value}")
    print('test')
    kpi_results2 = KPIs('E:/project/TimeList/data4/step2.5/test.csv',
                        'E:/project/TimeList/data4/label/test.csv')
    for key, value in kpi_results2.items():
        print(f"{key}: {value}")
    tp = kpi_results1['true_positive_length'] + kpi_results2['true_positive_length']
    tn = kpi_results1['true_negative_length'] + kpi_results2['true_negative_length']
    fp = kpi_results1['false_positive_length'] + kpi_results2['false_positive_length']
    fn = kpi_results1['false_negative_length'] + kpi_results2['false_negative_length']
    total = tp + tn + fp + fn
    precision = tp / (
            tp + fp) if tp > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    accuracy = (tp + tn) / total if total > 0 else 0
    print(tp, tn, fp, fn)
    print(precision, recall, f1_score, accuracy)