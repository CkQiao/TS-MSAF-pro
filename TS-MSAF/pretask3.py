import os
import pandas as pd


def find_consecutive_ones(numbers):
    # 初始化结果列表
    intervals = []
    start = None

    # 遍历数字列表
    for i, num in enumerate(numbers):
        if num == 1:
            # 如果遇到1，并且start还未设置，则设置start
            if start is None:
                start = i
        elif start is not None:
            # 如果遇到0，并且start已设置，则记录区间并重置start
            intervals.append((start, i - 1))
            start = None

            # 检查最后一个区间（如果最后一个数字是1）
    if start is not None:
        intervals.append((start, len(numbers) - 1))

    return intervals


def convert_anomalies_to_csv(input_dir, output_base_dir):
    # 遍历文件夹及其子文件夹
    for root, dirs, files in os.walk(input_dir):
        # 提取当前子文件夹名（不带路径）
        subfolder_name = os.path.basename(root)
        if not subfolder_name == 'test':
            continue
        # 创建输出CSV的文件名（使用子文件夹名）
        output_csv_path = os.path.join(output_base_dir, f"{subfolder_name}.csv")

        # 初始化DataFrame来存储所有标签数据
        all_intervals = []

        # 遍历当前子文件夹下的所有.txt文件
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)

                # 读取文件并解析每行数据（假设每行一个数字）
                with open(file_path, 'r') as f:
                    numbers = [int(line.strip()) for line in f]

                    # 找到连续1的区间
                intervals = find_consecutive_ones(numbers)

                # 为每个区间创建数据条目，使用文件名作为trace_name
                trace_name = os.path.splitext(file)[0]
                for start, end in intervals:
                    all_intervals.append({
                        'trace_name': trace_name,
                        'root_cause_start': start,
                        'root_cause_end': end,
                    })

                    # 将列表转换为DataFrame
        df = pd.DataFrame(all_intervals)

        # 将DataFrame保存为CSV文件
        df.to_csv(output_csv_path, index=False)

    # 使用函数，替换为你的输入和输出目录


input_dir = 'E:/project/TimeList/data4/label'
output_base_dir = 'E:/project/TimeList/data4/label'
convert_anomalies_to_csv(input_dir, output_base_dir)