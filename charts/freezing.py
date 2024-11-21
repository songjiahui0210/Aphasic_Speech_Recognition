import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams,font_manager

# Set the font to Poppins
font_path = 'fonts/Poppins-Medium.ttf'
font_manager.fontManager.addfont(font_path)
rcParams['font.family'] = 'Poppins'
rcParams['font.weight'] = 'medium'

fontsize_title=22
fontsize_label=22
fontsize_ticks=20

# Data
encoder_layers = [
    0, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 
    21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32
]
wer_values = [
    15.84, 16.38, 16.46, 16.85, 18.18, None, 17.47, 16.49, 17.8, 18.18, 
    18.38, 15.93, 18.33, 18.13, 16.16, 19.08, 17.51, 17.72, 17.25, 
    17.70, 19.31, 17.70, 18.96, 18.51
]

# Filter valid data points
valid_layers = [layer for layer, wer in zip(encoder_layers, wer_values) if wer is not None]
valid_wer = [wer for wer in wer_values if wer is not None]

# Plot
plt.figure(figsize=(10, 5))

# Plot the first dot in red
plt.scatter(valid_layers[0], valid_wer[0], color="#C8102E", s=60, label="First Point")
plt.text(valid_layers[0]+0.5, valid_wer[0], f'{valid_wer[0]:.2f}', fontsize=fontsize_ticks, ha='left', color="#C8102E")

# Plot the lowest dot in gold
plt.scatter(valid_layers[10], valid_wer[10], color="#A4804A", s=60, label="lowest Point")
plt.text(valid_layers[10]+0.1, valid_wer[10]+0.3, f'{valid_wer[10]:.2f}', fontsize=fontsize_ticks, ha='left', color="#A4804A")
plt.text(valid_layers[10]+3.5, valid_wer[10]-0.8, f'Layers 1 to {valid_layers[10]} frozen', fontsize=fontsize_ticks, ha='right', color="#A4804A")

# Plot the rest of the dots in black
other_layers = valid_layers[1:10] + valid_layers[11:]
other_wer = valid_wer[1:10] + valid_wer[11:]
plt.scatter(other_layers, other_wer, color="#000000", s=60, label="Other Points")

plt.title("WER (%) for Different Number of Bottom Layers Frozen", fontsize=fontsize_title)
plt.xlabel("Number of Layers Frozen", fontsize=fontsize_label)
plt.ylabel("WER (%)", fontsize=fontsize_label)
# plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(range(0, 33, 5)) 
plt.yticks(range(12, 22, 2))
plt.xticks(fontsize=fontsize_ticks)
plt.yticks(fontsize=fontsize_ticks)
# plt.xlim(left=0)  

# Remove upper and right borders
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()

# Save and show
# plt.savefig("encoder_freezing_dot_chart.png")
plt.show()