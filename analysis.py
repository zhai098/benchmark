import os
import json
import jsonlines
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import re

# 读取 JSONL 文件并修复格式
file_path = "/home/zhaipengxiang/111/outputs/ibm-granite/granite-4.0-micro_openai/gpt-oss-20b_862536/case_results.jsonl"

# 尝试加载 JSONL 文件并修复其中的格式问题
data = []
def fix_json_format(json_string):
    # 修复可能的单引号问题，将 JSON 键包裹在双引号内
    fixed_json = re.sub(r"'([^']+)':", r'"\1":', json_string)
    return fixed_json

try:
    with jsonlines.open(file_path) as reader:
        for line in reader:
            json_string = json.dumps(line)  # 转换为JSON字符串
            fixed_json = fix_json_format(json_string)  # 修复格式
            data.append(json.loads(fixed_json))  # 重新加载为Python对象
except Exception as e:
    print(f"Error reading the JSONL file: {e}")
    data = []

# 将 data 转换为 DataFrame
data_df = pd.DataFrame(data)

# 获取前25个案例
data_subset = data_df.head(25)

# 创建保存分析结果的文件夹
analysis_dir = os.path.abspath("./analysis")
analysis_dir = os.path.join(analysis_dir, "granite-4.0-micro")
os.makedirs(analysis_dir, exist_ok=True)

# 计算前25个case的分数平均值和方差
avg_score = data_subset['score'].mean()
score_variance = data_subset['score'].var()
avg_difficulty = data_subset['difficulty'].mean()
difficulty_variance = data_subset['difficulty'].var()

# 打印和保存统计信息
stats_file_path = os.path.join(analysis_dir, 'stats.txt')
with open(stats_file_path, 'w') as f:
    f.write(f"Average score for the first 25 cases: {avg_score}\n")
    f.write(f"Score variance for the first 25 cases: {score_variance}\n")
    f.write(f"Average difficulty for the first 25 cases: {avg_difficulty}\n")
    f.write(f"Difficulty variance for the first 25 cases: {difficulty_variance}\n")

print(f"Average score: {avg_score}")
print(f"Score variance: {score_variance}")
print(f"Average difficulty: {avg_difficulty}")
print(f"Difficulty variance: {difficulty_variance}")

# 设置seaborn主题
sns.set(style="whitegrid")

# 创建更精美的图表，使用seaborn来绘制

# 分数与难度关系（添加回归线和置信区间）
score_vs_difficulty_path = os.path.join(analysis_dir, 'score_vs_difficulty.png')
plt.figure(figsize=(12, 9))
sns.regplot(data=data_subset, x='difficulty', y='score', scatter_kws={'s': 100, 'color': 'purple', 'edgecolor': 'black', 'alpha': 0.7}, line_kws={'color': 'red'}, ci=95)
plt.title('Score vs Difficulty for the First 25 Cases', fontsize=18)
plt.xlabel('Difficulty', fontsize=14)
plt.ylabel('Score', fontsize=14)
plt.savefig(score_vs_difficulty_path)
plt.close()

# 分数与步骤数量关系（使用不同颜色，添加回归线）
score_vs_steps_path = os.path.join(analysis_dir, 'score_vs_steps.png')
plt.figure(figsize=(12, 9))
sns.regplot(data=data_subset, x='num_steps', y='score', scatter_kws={'s': 100, 'color': 'darkblue', 'edgecolor': 'black', 'alpha': 0.7}, line_kws={'color': 'green'}, ci=95)
plt.title('Score vs Number of Steps for the First 25 Cases', fontsize=18)
plt.xlabel('Number of Steps', fontsize=14)
plt.ylabel('Score', fontsize=14)
plt.savefig(score_vs_steps_path)
plt.close()

# 分数分布（使用色彩渐变和分布显示）
score_distribution_path = os.path.join(analysis_dir, 'score_distribution.png')
plt.figure(figsize=(12, 9))
sns.histplot(data_subset['score'], kde=True, color='green', bins=10, line_kws={'color': 'black', 'linewidth': 2})
plt.title('Score Distribution for the First 25 Cases', fontsize=18)
plt.xlabel('Score', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.savefig(score_distribution_path)
plt.close()

# 难度分布（色彩渐变）
difficulty_distribution_path = os.path.join(analysis_dir, 'difficulty_distribution.png')
plt.figure(figsize=(12, 9))
sns.histplot(data_subset['difficulty'], kde=True, color='orange', bins=10, line_kws={'color': 'black', 'linewidth': 2})
plt.title('Difficulty Distribution for the First 25 Cases', fontsize=18)
plt.xlabel('Difficulty', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.savefig(difficulty_distribution_path)
plt.close()

# 输出保存路径
(score_vs_difficulty_path, score_vs_steps_path, score_distribution_path, difficulty_distribution_path, stats_file_path)
