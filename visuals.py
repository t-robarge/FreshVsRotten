import matplotlib.pyplot as plt
import numpy as np
from data import FruitDataset, resnet_transform, load_data
# visualize


df = load_data('dataset/Train')
dataset = FruitDataset(df, transform=resnet_transform)

# Invert the fruit_to_idx mapping for easy lookup: idx -> fruit_name
idx_to_fruit = {v: k for k, v in dataset.fruit_to_idx.items()}

# Initialize counts: { "apple": {0: count_fresh, 1: count_rotten}, ... }
counts = {}
for fruit_name in dataset.fruit_to_idx:
    counts[fruit_name] = {0: 0, 1: 0}  # 0=fresh, 1=rotten

# Iterate over the dataset to fill in counts
for i in range(len(dataset)):
    _, fruit_label, rotten_label = dataset[i]
    fruit_name = idx_to_fruit[fruit_label.item()]
    counts[fruit_name][rotten_label.item()] += 1
fruit_names = sorted(counts.keys())  # e.g. ['apple', 'banana', 'orange']
fresh_counts = [counts[f][0] for f in fruit_names]  # 0 => fresh
rotten_counts = [counts[f][1] for f in fruit_names] # 1 => rotten

x = np.arange(len(fruit_names))  # the label locations
width = 0.35                     # width of the bars

fig, ax = plt.subplots(figsize=(8, 5))

# Plot fresh bars
rects1 = ax.bar(x - width/2, fresh_counts, width, label='Fresh')
# Plot rotten bars
rects2 = ax.bar(x + width/2, rotten_counts, width, label='Rotten')

# Add labels, title, and custom x-axis tick labels, etc.
ax.set_ylabel('Count')
ax.set_title('Fruit Distribution by Fresh/Rotten')
ax.set_xticks(x)
ax.set_xticklabels(fruit_names, rotation=45)
ax.legend()

# Optionally display the counts on top of each bar
for rects in [rects1, rects2]:
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

plt.tight_layout()
plt.show()