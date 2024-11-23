import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams,font_manager

# Set the font to Times New Roman
rcParams['font.family'] = 'Times New Roman'
rcParams['font.size'] = 8

# Data
layers = list(range(1, 33))
cka_similarity = [
    0.9999, 0.9982, 0.9980, 0.9967, 0.9950, 0.9892, 0.9805, 0.9793,
    0.9771, 0.9760, 0.9757, 0.9745, 0.9737, 0.9730, 0.9730, 0.9708,
    0.9702, 0.9697, 0.9679, 0.9660, 0.9640, 0.5483, 0.5483, 0.5482,
    0.5482, 0.5481, 0.5482, 0.5481, 0.5482, 0.5484, 0.5483, 0.5484
]

plt.figure(figsize=(6, 3))
plt.scatter(layers, cka_similarity, color="#000000", s=20, label="CKA Similarity")
plt.plot(layers, cka_similarity, color="#000000", linestyle='-', linewidth=0.6)


# Labels and title
# plt.title("Layer-wise CKA Similarity: Original vs Fine-tuned", fontsize=14)
plt.xlabel("Layer", fontsize=12)
plt.ylabel("CKA Similarity", fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
# plt.legend(loc='upper right', fontsize=12)

# Remove upper and right borders
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()