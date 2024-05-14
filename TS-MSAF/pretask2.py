import os
import pandas as pd


def convert_labels_to_csv(input_dir, output_base_dir):
    # 遍历文件夹及其子文件夹
    for root, dirs, files in os.walk(input_dir):
        # 提取当前子文件夹名（不带路径）
        subfolder_name = os.path.basename(root)
        if not subfolder_name == 'train':
            continue
        # 创建输出CSV的文件名（使用子文件夹名）
        output_csv_path = os.path.join(output_base_dir, f"{subfolder_name}.csv")

        # 初始化DataFrame来存储所有标签数据
        all_labels = []

        # 遍历当前子文件夹下的所有.txt文件
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)

                # 读取文件并解析每行数据
                with open(file_path, 'r') as f:
                    for line in f:
                        # 去除行尾的换行符并分割字符串
                        parts = line.strip().split(':')

                        # 检查是否有两部分（时间范围和标签列表）
                        if len(parts) == 2:
                            time_range, labels_str = parts

                            # 分割时间范围字符串并转换为整数
                            start, end = map(int, time_range.split('-'))

                            # 获取文件名（不带扩展名）作为trace_name
                            trace_name = os.path.splitext(file)[0]

                            # 创建一个字典来存储这一行的数据
                            label_data = {
                                'trace_name': trace_name,
                                'root_cause_start': start,
                                'root_cause_end': end,
                            }

                            # 将字典添加到列表中
                            all_labels.append(label_data)

                            # 将列表转换为DataFrame
        df = pd.DataFrame(all_labels)

        # 将DataFrame保存为CSV文件
        df.to_csv(output_csv_path, index=False)

    # 使用函数，替换为你的输入和输出目录


input_dir = 'E:/project/TimeList/data4/label'
output_base_dir = 'E:/project/TimeList/data4/label'
convert_labels_to_csv(input_dir, output_base_dir)