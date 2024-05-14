import os
import pandas as pd
import numpy as np


# 定义函数计算相对方差
def relative_variance(series):
    std_dev = series.dropna().std()
    mean = series.dropna().mean()
    return mean, std_dev


# 定义函数处理文件夹中的CSV文件并计算指标相对方差
def process_folder(folder_path):
    # 初始化指标相对方差字典
    stds = {}
    means = {}
    file_count = 0  # 记录文件数

    # 遍历文件夹中的CSV文件
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv') and filename != 'ground_truth.csv':
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path, index_col='t')
            df.drop(df.head(11).index, inplace=True)  # 从头去掉n行
            df.drop(df.tail(10).index, inplace=True)  # 从尾部去掉n行
            # 遍历每个指标
            for column in df.columns:
                try:
                    mean, std = relative_variance(df[column])
                    if column not in stds:
                        stds[column] = []
                    if column not in means:
                        means[column] = []
                        # 将相对方差添加到对应指标的列表中
                    if not np.isnan(mean) and not np.isnan(std):
                        stds[column].append(std)
                        means[column].append(mean)
                except Exception as e:
                    # 打印错误并继续处理其他文件或指标
                    print(f"Error calculating relative variance for {column} in {filename}: {e}")

            file_count += 1  # 增加文件计数
    # 返回指标相对方差字典和文件计数
    return stds, means


# 处理数据集并保存结果
def process_dataset(dataset_root, output_dir):
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

        # 遍历数据集根目录下的所有文件夹
    for foldername in os.listdir(dataset_root):
        folder_path = os.path.join(dataset_root, foldername)
        if os.path.isdir(folder_path):
            # 处理文件夹中的文件
            std, means = process_folder(folder_path)
            # 构造输出文件名
            output_filename = f"{foldername}.csv"
            output_file_path = os.path.join(output_dir, output_filename)

            # 将结果保存到CSV文件
            # 使用字典的items()方法将字典转换为列表，并创建DataFrame
            df_std = pd.DataFrame(list(std.items()), columns=['指标名', '标准差'])
            df_means = pd.DataFrame(list(means.items()), columns=['指标名', '均值'])

            # 使用pandas的merge方法按照'指标名'合并两个DataFrame
            output_df = pd.merge(df_std, df_means, on='指标名')

            output_df.to_csv(output_file_path, index=False)
            print(f"处理完成并保存结果到 {output_file_path}")

        # 设置数据集根目录和输出目录


dataset_root = 'E:/project/TimeList/data/'
output_dir = '.'  # 当前目录

# 调用函数处理数据集
process_dataset(dataset_root, output_dir)
