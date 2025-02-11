import matplotlib.pyplot as plt
import numpy as np

# Data for the bar graph
applications = ["Discord", "FB Messenger", "MS Teams", "Others", "Signal", "Telegram", "Whatsapp"]
cnn_accuracies = [91.84, 94.51, 92.14, 98.29, 86.86, 93.16, 99]
lstm_accuracies = [86.04, 85.8, 80.31, 88.14, 80.96, 75.09, 98.35]

# Plotting
plt.figure(figsize=(12, 8))
bar_width = 0.15  # Width of bars
index = np.arange(len(applications))

# Bars for each model
plt.barh(index + 2 * bar_width, cnn_accuracies, bar_width, label='CNN', color='lightgreen')
plt.barh(index + 3 * bar_width, lstm_accuracies, bar_width, label='LSTM', color='purple')

# Labels, Title, and Legend
plt.xlabel('Accuracy')
plt.title('Comparison of Accuracy for CNN and LSTM')
plt.yticks(index + 2 * bar_width, applications)
plt.legend()

# Adding grid lines at 60% and 80%
plt.axvline(x=60, color='grey', linestyle='--', linewidth=0.7)
plt.axvline(x=80, color='grey', linestyle='--', linewidth=0.7)
plt.axvline(x=90, color='grey', linestyle='--', linewidth=0.7)
plt.axvline(x=100, color='grey', linestyle='--', linewidth=0.7)

plt.tight_layout()
plt.show()
