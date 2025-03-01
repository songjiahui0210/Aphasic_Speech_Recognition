import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('../data_processed/dataset_splitted_by_duration.csv')

# 1. 统计每个 split 的说话者分布
speaker_counts = df.groupby('split')['name_unique_speaker'].nunique()
print("说话者数量分布：")
print(speaker_counts)

# 2. 计算每个 split 的总时长
duration_per_split = df.groupby('split')['duration'].sum()
print("\n各数据集总时长（秒）：")
print(duration_per_split)

# 3. 计算每个说话者在各数据集中的时长
speaker_duration_split = df.groupby(['name_unique_speaker', 'split'])['duration'].sum().unstack(fill_value=0)
print("\n每个说话者在不同数据集中的时长：")
print(speaker_duration_split)

# 4. 确认说话者唯一性
train_speakers = set(df[df['split'] == 'train']['name_unique_speaker'])
test_speakers = set(df[df['split'] == 'test']['name_unique_speaker'])
eval_speakers = set(df[df['split'] == 'eval']['name_unique_speaker'])

assert train_speakers.isdisjoint(test_speakers), "错误: 一些说话者同时出现在 train 和 test 中！"
assert train_speakers.isdisjoint(eval_speakers), "错误: 一些说话者同时出现在 train 和 eval 中！"
assert test_speakers.isdisjoint(eval_speakers), "错误: 一些说话者同时出现在 test 和 eval 中！"
print("\n说话者唯一性检查通过！")

# 5. 可视化
plt.figure(figsize=(8, 5))
plt.pie(speaker_counts, labels=speaker_counts.index, autopct='%1.1f%%', startangle=90)
plt.title('说话者数量在不同数据集中的比例')
plt.show()

plt.figure(figsize=(8, 5))
plt.pie(duration_per_split, labels=duration_per_split.index, autopct='%1.1f%%', startangle=90)
plt.title('音频时长在不同数据集中的比例')
plt.show()