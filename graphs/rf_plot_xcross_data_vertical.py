import matplotlib.pyplot as plt
import numpy as np

# Data for the bar graph
applications = ["Discord", "FB Messenger", "MS Teams", "Others", "Signal", "Telegram", "Whatsapp"]
same_data = [91.98, 98.14, 88.97, 91.59, 99.14, 96.15, 86.79]
xcross_data = [44.26, 32.87, 55.87, 45.04, 89.28, 54.84, 64.71]

# Plotting
plt.figure(figsize=(12, 8))
bar_width = 0.35  # Width of bars
index = np.arange(len(applications))

# Bars for each data set
same_bars = plt.bar(index - bar_width / 2, same_data, bar_width, label='Same Data', color='grey')
xcross_bars = plt.bar(index + bar_width / 2, xcross_data, bar_width, label='XCross Data', color='lightgreen')

# Labels, Title, and Legend
plt.ylabel('Accuracy - F1 Score (%)')
plt.title('Accuracy of RF on Same Data vs XCross Data')
plt.xticks(index, applications)
plt.legend(loc='lower right')  # Positioning the legend at the bottom right

# Adding grid lines at specific levels
plt.axhline(y=50, color='grey', linestyle='--', linewidth=0.7)
plt.axhline(y=75, color='grey', linestyle='--', linewidth=0.7)
plt.axhline(y=100, color='grey', linestyle='--', linewidth=0.7)

# Adding values with percentage sign on top of each bar
for bar in same_bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval + 1, f'{yval:.2f}%', ha='center', va='bottom')

for bar in xcross_bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval + 1, f'{yval:.2f}%', ha='center', va='bottom')

plt.tight_layout()
plt.show()

