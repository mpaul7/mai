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

mlp_lstm_f1_scores = [53.88, 44.19, 61.94, 59.38, 49.78, 52.37, 50.81]
mlp_cnn_f1_scores = [72.76, 61.12, 83.78, 72.87, 67.00, 60.51, 58.14]
lstm_cnn_f1_scores = [84.08, 77.92, 93.98, 86.50, 77.86, 70.95, 73.95]

# Plotting
plt.figure(figsize=(16, 8))
bar_width = 0.25  # Width of bars
index = np.arange(len(classes))

# Bars for each model
mlp_lstm_bars = plt.bar(index - bar_width, mlp_lstm_f1_scores, bar_width, label="MLP-LSTM", color="skyblue")
mlp_cnn_bars = plt.bar(index, mlp_cnn_f1_scores, bar_width, label="MLP-CNN", color="orange")
lstm_cnn_bars = plt.bar(index + bar_width, lstm_cnn_f1_scores, bar_width, label="LSTM-CNN", color="lightgreen")

# Labels, Title, and Legend
plt.xlabel("Applications", fontsize=14)
plt.ylabel("F1 Score (%)", fontsize=14)
plt.title("Multi-Head Model Performance (F1 Score)", fontsize=16)
plt.xticks(index, classes, fontsize=12, 
        #    rotation=45
           )
plt.legend(loc="upper right", fontsize=12)

# Adding grid lines
plt.axhline(y=50, color="grey", linestyle="--", linewidth=0.7)
plt.axhline(y=75, color="grey", linestyle="--", linewidth=0.7)

# Adding values with percentage sign on top of each bar
for bars in [mlp_lstm_bars, mlp_cnn_bars, lstm_cnn_bars]:
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 1, f"{yval:.2f}", ha="center", fontsize=10)

plt.tight_layout()
plt.show()
plt.savefig('/home/mpaul/projects/mpaul/mai/src_3/scripts/multi-head_bargraph.png')
