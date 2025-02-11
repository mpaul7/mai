import matplotlib.pyplot as plt

# Data for the bar graph
applications = ["Bitcoin", "Others"]
mlp_accuracies = [89.77, 90.6]
lstm_accuracies = [98.42, 98.41]
xgboost_accuracies = [95.82, 95.73]

# Plotting
plt.figure(figsize=(10, 6))
bar_width = 0.35  # Width of bars
index = range(len(applications))

# Bars for RF and XGBoost
plt.barh(index, rf_accuracies, bar_width, label='RF', color='grey')
plt.barh([i + bar_width for i in index], xgboost_accuracies, bar_width, label='XGBoost', color='salmon')

# Labels, Title, and Legend
plt.ylabel('Accuracy')
plt.title('Comparison of Accuracy for RF and XGBoost')
plt.xticks([i + bar_width / 2 for i in index], applications)
plt.legend()

# Adding grid lines at 60% and 80%
plt.axhline(x=60, color='grey', linestyle='--', linewidth=0.7)
plt.axhline(x=80, color='grey', linestyle='--', linewidth=0.7)

plt.tight_layout()
plt.show()
