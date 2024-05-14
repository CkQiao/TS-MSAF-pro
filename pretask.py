import os
import pandas as pd


def convert_txt_to_csv(input_dir, output_dir):
    # 遍历文件夹及其子文件夹
    for root, dirs, files in os.walk(input_dir):
        # 构造输出目录的相对路径
        rel_path = os.path.relpath(root, input_dir)
        output_root = os.path.join(output_dir, rel_path)

        # 如果输出目录不存在，则创建它
        if not os.path.exists(output_root):
            os.makedirs(output_root)

        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                # 读取txt文件到DataFrame
                df = pd.read_csv(file_path, header=None, delimiter=',', names=None)

                # 添加索引列't'
                df.reset_index(inplace=True)
                df.columns = ['t'] + ['metric{}'.format(i) for i in range(1, len(df.columns))]

                # 构造输出的csv文件路径
                output_file_path = os.path.join(output_root, os.path.splitext(file)[0] + '.csv')

                # 保存为csv文件
                df.to_csv(output_file_path, index=False)

            # 使用函数，替换为你的输入和输出目录


input_dir = 'E:/project/TimeList/data4/raw'
output_dir = 'E:/project/TimeList/data4/original'
convert_txt_to_csv(input_dir, output_dir)