import matplotlib.pyplot as plt
import numpy as np

# Data for the bar graph
classes = [
    "Discord",
    "Others",
    "Telegram",
    "MS Teams",
    "WhatsApp",
    "FB Messenger",
    "Signal",
]

mlp_f1_scores = [53.01, 36.14, 23.58, 55.52, 61.59, 52.63, 44.72]
lstm_seq_f1_scores = [80.56, 68.86, 87.26, 85.32, 76.5, 61.86, 65.86]
cnn_seq_f1_scores = [78.14, 68.42, 90.19, 69.57, 76.16, 59.51, 59.98]
cnn_stat_f1_scores = [82.06, 71.74, 92.56, 83.55, 75.89, 68.54, 74.82]

# Plotting
plt.figure(figsize=(16, 8))
bar_width = 0.2  # Width of bars
index = np.arange(len(classes))

# Bars for each model
mlp_bars = plt.bar(index - 1.5 * bar_width, mlp_f1_scores, bar_width, label="MLP-Stat", color="skyblue")
lstm_bars = plt.bar(index - 0.5 * bar_width, lstm_seq_f1_scores, bar_width, label="LSTM-Seq", color="orange")
cnn_seq_bars = plt.bar(index + 0.5 * bar_width, cnn_seq_f1_scores, bar_width, label="CNN-Seq", color="lightgreen")
cnn_stat_bars = plt.bar(index + 1.5 * bar_width, cnn_stat_f1_scores, bar_width, label="CNN-Stat", color="salmon")

# Labels, Title, and Legend
plt.xlabel("Applications", fontsize=14)
plt.ylabel("F1 Score (%)", fontsize=14)
plt.title("Single-Head Model Performance (F1 Score)", fontsize=16)
plt.xticks(index, classes, fontsize=12, 
        #    rotation=45
           )
plt.legend(loc="upper right", fontsize=12)

# Adding grid lines
plt.axhline(y=50, color="grey", linestyle="--", linewidth=0.7)
plt.axhline(y=75, color="grey", linestyle="--", linewidth=0.7)

# Adding values with percentage sign on top of each bar
for bars in [mlp_bars, lstm_bars, cnn_seq_bars, cnn_stat_bars]:
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 1, f"{yval:.2f}", ha="center", fontsize=10)

plt.tight_layout()
plt.show()
plt.savefig('/home/mpaul/projects/mpaul/mai/src_3/scripts/single-head_bargraph.png')
