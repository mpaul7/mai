import matplotlib.pyplot as plt
import numpy as np

# Data for the bar graph
applications = ["Discord", "Others", "Telegram", "Microsoft Teams", "Whatsapp", "FB Messenger", "Signal"]
# lstm_accuracies = [80.56, 68.86, 87.26, 85.32, 76.5, 61.86, 65.86]
# cnn_seq_accuracies = [78.14, 68.42, 90.19, 69.57, 76.16, 59.51, 59.98]
cnn_stat_accuracies = [82.06, 71.74, 92.56, 83.55, 75.89, 68.54, 74.82]
lstm_cnn_stat_accuracies = [84.08, 77.92, 93.98, 86.50, 77.86, 70.95, 73.95]

# Plotting
plt.figure(figsize=(16, 8))
bar_width = 0.2  # Width of bars
index = np.arange(len(applications))

# Bars for each model
# lstm_bars = plt.bar(index - 1.5 * bar_width, lstm_accuracies, bar_width, label='LSTM', color='lightblue')
# cnn_seq_bars = plt.bar(index - 0.5 * bar_width, cnn_seq_accuracies, bar_width, label='CNN-Seq', color='olive')
cnn_stat_bars = plt.bar(index + 0.5 * bar_width, cnn_stat_accuracies, bar_width, label='CNN-Stat', color='lightgreen')
lstm_cnn_stat_bars = plt.bar(index + 1.5 * bar_width, lstm_cnn_stat_accuracies, bar_width, label='LSTM-CNN-Stat', color='orange')

# Labels, Title, and Legend
plt.ylabel('Accuracy (%)', fontsize=14)
plt.title('Comparison of Single-head and Multi-head Model Accuracies Across Applications', fontsize=14)
plt.xticks(index, applications, fontsize=14)
plt.legend(loc='upper right', fontsize=16)  # Positioning the legend at the bottom right

# Adding grid lines for better readability
plt.axhline(y=50, color='grey', linestyle='--', linewidth=0.7)
plt.axhline(y=75, color='grey', linestyle='--', linewidth=0.7)
plt.axhline(y=90, color='grey', linestyle='--', linewidth=0.7)

# Adding values with percentage sign on top of each bar
for bars in [
    # lstm_bars, cnn_seq_bars, 
             cnn_stat_bars, lstm_cnn_stat_bars]:
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 1, f'{yval:.2f}', ha='center', va='bottom'
                #  , rotation=45
                 )

plt.tight_layout()
plt.show()
plt.savefig('/home/mpaul/projects/mpaul/mai/results/results_jan13/cross_dataset_v2/graphs/single_multi_head_comparison_V2.png')
