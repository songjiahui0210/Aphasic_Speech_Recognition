import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams,font_manager

# Set the font to Poppins
font_path = 'fonts/Poppins-Medium.ttf'
font_manager.fontManager.addfont(font_path)
rcParams['font.family'] = 'Poppins'
rcParams['font.weight'] = 'medium'

# Dataset 1: Overall distribution
overall_labels = ['Training', 'Validation', 'Test']
overall_sizes = [77.35, 11.81, 10.84]
overall_colors = ['#609F80', '#E2A855', '#BB4100']

# Dataset 2: Training set distribution
training_labels = ['Mild', 'Moderate', 'Severe', 'Very Severe', 'Unknown']
training_sizes = [55.41, 19.35, 4.67, 0.95, 19.69]
training_colors = ['#98D1B5', '#FFD580', '#FF9966', '#A0E0EF', '#A4804A']

# Create figure and axes
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Overall distribution pie chart
wedges, texts, autotexts = axs[0].pie(
    overall_sizes,
    labels=overall_labels,
    autopct='%1.1f%%',
    startangle=30,
    colors=overall_colors,
    wedgeprops={'edgecolor': 'white'},
    textprops={'fontsize': 12} 
)
axs[0].set_title('Overall Dataset Distribution', fontsize=14)

# Training set distribution pie chart
axs[1].pie(
    training_sizes,
    labels=training_labels,
    autopct='%1.1f%%',
    startangle=140,
    colors=training_colors,
    wedgeprops={'edgecolor': 'white'},
    textprops={'fontsize': 12} 
)
axs[1].set_title('Severity Level Distribution', fontsize=14)

plt.tight_layout()
plt.show()