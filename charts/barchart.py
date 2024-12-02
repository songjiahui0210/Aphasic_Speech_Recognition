import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams,font_manager

# Set the font to Poppins
font_path = 'fonts/Poppins-Medium.ttf'
font_manager.fontManager.addfont(font_path)
rcParams['font.family'] = 'Poppins'
rcParams['font.weight'] = 'medium'

# Data from the table
models = ["Small", "Medium", "Large-v3"]
baseline_wer = [51.15, 37.79, 27.16]
fine_tuned_wer = [22.73, 17.18, 15.84]

# Bar width and positions
bar_width = 0.35
x = np.arange(len(models))

# Create the bar chart
fig, ax = plt.subplots(figsize=(8, 6))
bars1 = ax.bar(x - bar_width / 2, baseline_wer, bar_width, label="Baseline WER", color="#E2A855")
bars2 = ax.bar(x + bar_width / 2, fine_tuned_wer, bar_width, label="Fine-Tuned WER", color="#025944")

# Add labels, title, and legend
# ax.set_xlabel("Models", fontsize=12)
ax.set_ylabel("WER (%)", fontsize=12)
# ax.set_title("Baseline vs. Fine-Tuned WER for Whisper Models", fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=11)
ax.legend(fontsize=11)

# Add values on top of the bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height, f"{height:.2f}", ha="center", va="bottom", fontsize=10)

plt.tight_layout()
plt.show()