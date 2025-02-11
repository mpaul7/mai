import matplotlib.pyplot as plt
import numpy as np

# Data for the bar graph
applications = ["Discord", "FB Messenger", "MS Teams", "Others", "Signal", "Telegram", "Whatsapp"]
rf_accuracies = [91.98, 98.14, 88.97, 91.59, 99.14, 96.15, 86.79]
xgboost_accuracies = [92.38, 98.53, 89.68, 92.57, 99.3, 96.5, 87.3]

# Plotting
plt.figure(figsize=(12, 8))
bar_width = 0.30  # Width of bars
index = np.arange(len(applications))

# Bars for each model
rf_bars = plt.bar(index - bar_width / 2, rf_accuracies, bar_width, label='Random Forest', color='grey')
xgboost_bars = plt.bar(index + bar_width / 2, xgboost_accuracies, bar_width, label='XGBoost', color='salmon')

# Labels, Title, and Legend
plt.ylabel('Accuracy - F1 Score (%)')
plt.title('Comparison of Accuracy for Random Forest and XGBoost')
plt.xticks(index, applications)
plt.legend()

# Adding grid lines at specific levels
plt.axhline(y=70, color='grey', linestyle='--', linewidth=0.7)
plt.axhline(y=80, color='grey', linestyle='--', linewidth=0.7)
plt.axhline(y=90, color='grey', linestyle='--', linewidth=0.7)
plt.axhline(y=100, color='grey', linestyle='--', linewidth=0.7)

# Adding values with percentage sign on top of each bar
for bar in rf_bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval + 1, f'{yval:.2f}', ha='center', va='bottom')

for bar in xgboost_bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval + 1, f'{yval:.2f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

