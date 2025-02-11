import matplotlib.pyplot as plt
import numpy as np

# Data for the bar graph
applications = [
    "Discord",
    "Others",
    "Telegram",
    "Microsoft Teams",
    "WhatsApp",
    "Facebook Messenger",
    "Signal",
]
lstm_accuracies = [82.06, 71.74, 92.56, 83.55, 75.89, 68.54, 74.82]
lstm_cnn_stat_accuracies = [84.08, 77.92, 93.98, 86.50, 77.86, 70.95, 73.95]

cnn_stat_accuracies = [82.06, 71.74, 92.56, 83.55, 75.89, 68.54, 74.82]
lstm_cnn_stat_accuracies = [84.08, 77.92, 93.98, 86.50, 77.86, 70.95, 73.95]

# Plotting
plt.figure(figsize=(14, 8))
bar_width = 0.35  # Width of bars
index = np.arange(len(applications))

# Bars for each model
lstm_bars = plt.bar(index - bar_width / 2, lstm_accuracies, bar_width, label="CNN-Stat", color="orange")
lstm_cnn_stat_bars = plt.bar(index + bar_width / 2, lstm_cnn_stat_accuracies, bar_width, label="LSTM-CNN-Stat", color="lightgreen")

# Labels, Title, and Legend
plt.xlabel("Applications", fontsize=14)
plt.ylabel("F1 Score (%)", fontsize=14)
plt.title("Comparison of CNN-Stat and LSTM-CNN-Stat Performance", fontsize=16)
plt.xticks(index, applications, fontsize=12
        #    , rotation=45
           )
plt.legend(loc="upper right", fontsize=12)

# Adding grid lines
plt.axhline(y=50, color="grey", linestyle="--", linewidth=0.7)
plt.axhline(y=75, color="grey", linestyle="--", linewidth=0.7)

# Adding values with percentage sign on top of each bar
for bars in [lstm_bars, lstm_cnn_stat_bars]:
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 1, f"{yval:.2f}", ha="center", fontsize=10)

plt.tight_layout()
plt.show()
plt.savefig('/home/mpaul/projects/mpaul/mai/src_3/scripts/single-head_multi-head_bargraph.png')
