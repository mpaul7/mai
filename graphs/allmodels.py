import matplotlib.pyplot as plt

# Data for the bar graph
applications = ["Discord", "Facebook Messenger", "Microsoft Teams", "Others", "Signal", "Telegram", "Whatsapp"]
rf_accuracies = [91.98, 98.14, 88.97, 91.59, 99.14, 96.15, 86.79]
xgboost_accuracies = [92.38, 98.53, 89.68, 92.57, 99.3, 96.5, 87.3]
cnn_accuracies = [91.84, 94.51, 92.14, 98.29, 86.86, 93.16, 99]
lstm_accuracies = [86.04, 85.8, 80.31, 88.14, 80.96, 75.09, 98.35]

# Plotting
plt.figure(figsize=(12, 8))
bar_width = 0.2  # Width of bars
index = range(len(applications))

# Bars for each model
plt.barh(index, rf_accuracies, bar_width, label='RF', color='skyblue')
plt.barh([i + bar_width for i in index], xgboost_accuracies, bar_width, label='XGBoost', color='salmon')
plt.barh([i + 2 * bar_width for i in index], cnn_accuracies, bar_width, label='CNN', color='lightgreen')
plt.barh([i + 3 * bar_width for i in index], lstm_accuracies, bar_width, label='LSTM', color='purple')

# Labels, Title, and Legend
plt.xlabel('Accuracy')
plt.title('Comparison of Accuracy for RF, XGBoost, CNN, and LSTM')
plt.yticks([i + 1.5 * bar_width for i in index], applications)
plt.legend()

# Adding grid lines at 60% and 80%
plt.axvline(x=60, color='grey', linestyle='--', linewidth=0.7)
plt.axvline(x=80, color='grey', linestyle='--', linewidth=0.7)

plt.tight_layout()
plt.show()
