import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams,font_manager

# Set the font to Poppins
font_path = 'fonts/Poppins-Medium.ttf'
font_manager.fontManager.addfont(font_path)
rcParams['font.family'] = 'Poppins'
rcParams['font.weight'] = 'medium'

fontsize_title=26
fontsize_text=24

# Dataset 2: Training set distribution
labels = ['Mild', 'Moderate', 'Severe', 'Very Severe', 'Unknown']
sizes = [53.24, 21.84, 3.91, 0.59, 20.42]
colors = ['#C8102E', '#000000', '#A4804A', '#FF9999', '#666666']

# Create figure
plt.figure(figsize=(8, 8))

# Training set distribution pie chart
wedges, texts, autotexts = plt.pie(
    sizes,
    labels=labels,
    autopct='%1.1f%%',
    startangle=140,
    colors=colors,
    wedgeprops={'edgecolor': 'white'},
    textprops={'fontsize': fontsize_text}
)

# Set the color of the numbers to white
for autotext in autotexts:
    autotext.set_color('white')

# Adjust the position of the numbers and labels
index_unkown = labels.index('Unknown')
autotexts[index_unkown].set_position((autotexts[index_unkown].get_position()[0]-0.08, autotexts[index_unkown].get_position()[1] -0.08))

index_severe = labels.index('Severe')
autotexts[index_severe].set_position((autotexts[index_severe].get_position()[0]+0.1, autotexts[index_severe].get_position()[1]))
texts[index_severe].set_position((texts[index_severe].get_position()[0], texts[index_severe].get_position()[1] - 0.1))

index_very_severe = labels.index('Very Severe')
# autotexts[index_very_severe].set_color('black')
autotexts[index_very_severe].set_position((autotexts[index_very_severe].get_position()[0]-0.08, autotexts[index_very_severe].get_position()[1] + 0.12))
texts[index_very_severe].set_position((texts[index_very_severe].get_position()[0]-0.1, texts[index_very_severe].get_position()[1]))

plt.title('Severity Level Distribution', fontsize=fontsize_title)

# Adjust layout
plt.tight_layout()
# plt.savefig("training_set_distribution_pie_chart.png")
plt.show()