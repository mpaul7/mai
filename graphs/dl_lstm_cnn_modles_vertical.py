import matplotlib.pyplot as plt
import numpy as np

# Data for the bar graph
applications = ["Discord", "FB Messenger", "MS Teams", "Others", "Signal", "Telegram", "Whatsapp"]
cnn_accuracies = [91.84, 94.51, 92.14, 98.29, 86.86, 93.16, 99]
lstm_accuracies = [86.04, 85.8, 80.31, 88.14, 80.96, 75.09, 98.35]

# Plotting
plt.figure(figsize=(12, 8))
bar_width = 0.30  # Width of bars
index = np.arange(len(applications))

# Bars for each model
cnn_bars = plt.bar(index + 1 * bar_width, cnn_accuracies, bar_width, label='CNN', color='lightgreen')
lstm_bars = plt.bar(index + 2 * bar_width, lstm_accuracies, bar_width, label='LSTM', color='orange')

# Labels, Title, and Legend
plt.ylabel('Accuracy - F1 Score (%)')
plt.title('Comparison of Accuracy for CNN and LSTM')
plt.xticks(index + 2 * bar_width, applications)
plt.legend()

# Adding grid lines at specific levels
plt.axhline(y=70, color='grey', linestyle='--', linewidth=0.7)
plt.axhline(y=80, color='grey', linestyle='--', linewidth=0.7)
plt.axhline(y=90, color='grey', linestyle='--', linewidth=0.7)
plt.axhline(y=100, color='grey', linestyle='--', linewidth=0.7)

# Adding values on top of each bar
for bar in cnn_bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval + 1, f'{yval:.2f}%', ha='center', va='bottom')

for bar in lstm_bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval + 1, f'{yval:.2f}%', ha='center', va='bottom')

plt.tight_layout()
plt.show()

