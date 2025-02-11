import matplotlib.pyplot as plt
import numpy as np

# Data for the bar graph
applications = ["Discord", "FB Messenger", "MS Teams", "Others", "Signal", "Telegram", "Whatsapp"]
cnn_accuracies = [91.84, 94.51, 92.14, 98.29, 86.86, 93.16, 99]
lstm_accuracies = [86.04, 85.8, 80.31, 88.14, 80.96, 75.09, 98.35]
mlp_accuracies = [71.25, 56.42, 72.74, 74.9, 74.59, 55.86, 79.25]
rf_accuracies = [91.98, 98.14, 88.97, 91.59, 99.14, 96.15, 86.79]
xgboost_accuracies = [92.38, 98.53, 89.68, 92.57, 99.3, 96.5, 87.3]

# Plotting
plt.figure(figsize=(14, 8))
bar_width = 0.15  # Adjusted width to fit more bars
index = np.arange(len(applications))

# Bars for each model
cnn_bars = plt.bar(index + 1 * bar_width, cnn_accuracies, bar_width, label='CNN', color='lightgreen')
lstm_bars = plt.bar(index + 2 * bar_width, lstm_accuracies, bar_width, label='LSTM', color='orange')
mlp_bars = plt.bar(index + 3 * bar_width, mlp_accuracies, bar_width, label='MLP', color='skyblue')
rf_bars = plt.bar(index + 4 * bar_width, rf_accuracies, bar_width, label='Random Forest', color='grey')
xgboost_bars = plt.bar(index + 5 * bar_width, xgboost_accuracies, bar_width, label='XGBoost', color='salmon')

# Labels, Title, and Legend
plt.ylabel('Accuracy - F1 Score (%)')
plt.title('Comparison of Accuracy for CNN, LSTM, MLP, Random Forest, and XGBoost')
plt.xticks(index + 3 * bar_width, applications)
plt.legend(loc='lower right')

# Adding grid lines at specific levels
plt.axhline(y=60, color='grey', linestyle='--', linewidth=0.7)
plt.axhline(y=80, color='grey', linestyle='--', linewidth=0.7)
plt.axhline(y=90, color='grey', linestyle='--', linewidth=0.7)
plt.axhline(y=100, color='grey', linestyle='--', linewidth=0.7)

# Adding values with percentage sign on top of each bar

plt.tight_layout()
plt.show()

