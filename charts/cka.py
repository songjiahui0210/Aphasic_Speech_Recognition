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
layers = list(range(1, 33))
cka_similarity = [
    0.9999, 0.9982, 0.9980, 0.9967, 0.9950, 0.9892, 0.9805, 0.9793,
    0.9771, 0.9760, 0.9757, 0.9745, 0.9737, 0.9730, 0.9730, 0.9708,
    0.9702, 0.9697, 0.9679, 0.9660, 0.9640, 0.5483, 0.5483, 0.5482,
    0.5482, 0.5481, 0.5482, 0.5481, 0.5482, 0.5484, 0.5483, 0.5484
]

# Plot
plt.figure(figsize=(10, 5))
plt.scatter(layers, cka_similarity, color="#000000", s=60,label="CKA Similarity")
plt.plot(layers, cka_similarity, color="#000000", linestyle='-', linewidth=1)


# Labels and title
plt.title("Layer-wise CKA Similarity: Original vs Fine-tuned", fontsize=fontsize_title)
plt.xlabel("Layer", fontsize=fontsize_label)
plt.ylabel("CKA Similarity", fontsize=fontsize_label)
plt.xticks(fontsize=fontsize_ticks)
plt.yticks(fontsize=fontsize_ticks)
# plt.legend(loc='upper right', fontsize=12)

# Remove upper and right borders
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)


# Display
# plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()