import random

import matplotlib.pyplot as plt
import numpy as np

from KPI import KPIs
from allfiles import action

keys = [f'metric{i + 1}' for i in range(38)]
train_path = 'E:\project\TimeList/data4/step1/train'
train_label = 'E:\project\TimeList/data4/label/train.csv'
train_out = 'E:\project\TimeList/data4/step2/train.csv'
test_path = 'E:\project\TimeList/data4/step1/test'
test_label = 'E:\project\TimeList/data4/label/test.csv'
test_out = 'E:\project\TimeList/data4/step2/test.csv'

def KPI(weight, threshold, input_dir, output_file, real_file):
    action(weight, threshold, input_dir=input_dir, outputfile=output_file)
    kpi_results = KPIs(real_file=real_file, output_file=output_file)
    return kpi_results


# 遗传算法参数
def generate_individual(keys):
    return {key: random.choice([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]) for key
            in keys}


# 计算个体的适应度
def calculate_fitness(individual, threshold, real_file, output_file, input_dir):
    kpi_results = KPI(weight=individual, threshold=threshold, real_file=real_file, output_file=output_file,
                      input_dir=input_dir)
    return kpi_results['precision'] + kpi_results['recall'] + kpi_results['f1_score'] + kpi_results['accuracy']


# 选择
def select(population, fitnesses, num):
    indices = np.argsort(fitnesses)[-num:]
    return [population[i] for i in indices]


# 交叉
def crossover(parent1, parent2):
    child1, child2 = {}, {}
    keys = list(parent1.keys())
    split = random.randint(0, len(keys))
    for key in keys[:split]:
        child1[key] = parent1[key]
        child2[key] = parent2[key]
    for key in keys[split:]:
        child1[key] = parent2[key]
        child2[key] = parent1[key]
    return child1, child2


# 变异
def mutate(individual, mutation_rate=0.02):
    for key in individual.keys():
        if random.random() < mutation_rate:
            individual[key] = random.choice(
                [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5])
    return individual


fitness_list = []

fitls = []


# 遗传算法主函数
def genetic_algorithm(keys, threshold, population_size=100, generations=60, input_dir='./output',
                      output_file='./output.csv', real_file='./output.csv'):
    population = [generate_individual(keys) for _ in range(population_size)]
    for generation in range(generations):
        fitnesses = [
            calculate_fitness(individual, threshold, real_file=real_file, output_file=output_file, input_dir=input_dir)
            for individual in population]
        fits = [calculate_fitness(individual, threshold, input_dir=test_path, output_file=test_out,
                                  real_file=test_label) for individual in population]
        selected = select(population, fitnesses, population_size // 2)
        offspring = []
        while len(offspring) < population_size:
            parent1, parent2 = random.sample(selected, 2)
            child1, child2 = crossover(parent1, parent2)
            offspring.append(mutate(child1))
            if len(offspring) < population_size:
                offspring.append(mutate(child2))
        population = offspring
        fitness_list.append(max(fitnesses))
        fitls.append(max(fits))
        print(f"Generation {generation}: Max fitness = {max(fitnesses)}; Max fitness = {max(fits)}")
    return population


def plot_bar_chart(weight_dict):
    # 检查输入是否为字典类型
    if not isinstance(weight_dict, dict):
        raise ValueError("Input must be a dictionary.")

        # 检查字典是否为空
    if not weight_dict:
        raise ValueError("The dictionary cannot be empty.")

        # 获取字典的键和值
    labels = list(weight_dict.keys())
    values = list(weight_dict.values())

    # 确保所有的值都是数值类型
    if not all(isinstance(v, (int, float)) for v in values):
        raise ValueError("All values in the dictionary must be numeric.")

        # 绘制柱状图
    plt.figure(figsize=(10, 6))
    plt.bar(labels, values)
    plt.xlabel('Categories')
    plt.ylabel('Weights')
    plt.title('Bar Chart of Weight Distribution')
    plt.xticks(rotation=90)  # 如果标签过长，可以旋转以更好地显示
    plt.tight_layout()  # 调整布局以防止标签重叠
    plt.show()


# 运行遗传算法
best_population = genetic_algorithm(keys, threshold=3, population_size=100, generations=30,
                                    input_dir=train_path,
                                    output_file=train_out,
                                    real_file=train_label
                                    )

# 在所有生成的population中找到最好的解
best_fitness = -1
best_individual = None
for individual in best_population:
    fitness = calculate_fitness(individual, 3, input_dir=test_path,
                 output_file=test_out,
                 real_file=test_label)  # 假定的threshold值为5
    if fitness > best_fitness:
        best_fitness = fitness
        best_individual = individual

plt.figure()
plt.plot(fitness_list, label='train')
plt.plot(fitls, label='val')
plt.xlabel('generations')
plt.ylabel('fitness')
plt.legend(['train', 'val'])
plt.show()

print("Best Individual:", best_individual)
print("Best Fitness:", best_fitness)
plot_bar_chart(best_individual)
kpi_result = KPI(best_individual, 3, input_dir=test_path,
                 output_file=test_out,
                 real_file=test_label)
for key, value in kpi_result.items():
    print(f"{key}: {value}")
