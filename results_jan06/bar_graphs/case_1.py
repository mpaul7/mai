import matplotlib.pyplot as plt
import numpy as np

# Data for the two results
categories = ["discord", "others", "telegram", "microsoft_teams", "whatsapp", "facebook_messenger", "signal"]
models = ["1-mlp", "2-lstm", "3-cnn_seq", "4-cnn_stat"]

# First result
first_result = [
    [89.03, 90.27, 81.07, 83.69, 97.34, 82.11, 77.48],  # 1-mlp
    [92.48, 97.23, 90.94, 89.34, 98.83, 85.88, 78.64],  # 2-lstm
    [92.3, 97.77, 91.91, 90.75, 98.86, 86.88, 79.75],   # 3-cnn_seq
    [91.85, 90.36, 88.79, 65.06, 84.56, 63.82, 47.44]   # 4-cnn_stat
]

# Second result
second_result = [
    [86.62, 84.99, 74.62, 68.22, 97.14, 77.7, 60.13],   # 1-mlp
    [90.82, 96.46, 84.84, 80.25, 98.84, 83.11, 77.92],  # 2-lstm
    [90.8, 96.88, 68.36, 83, 98.84, 83.97, 75.69],      # 3-cnn_seq
    [89.94, 90.35, 82.64, 68.55, 68.78, 81.37, 45.17]   # 4-cnn_stat
]

# Bar width and positions
x = np.arange(len(categories))
bar_width = 0.2

# Create the plot
plt.figure(figsize=(15, 7))
for i, (model, first, second) in enumerate(zip(models, first_result, second_result)):
    plt.bar(x + i * bar_width, first, width=bar_width, label=f"{model} (First Result)")
    plt.bar(x + i * bar_width + bar_width/2, second, width=bar_width, label=f"{model} (Second Result)", alpha=0.7)

# Adding labels and formatting
plt.xlabel("Categories")
plt.ylabel("F1 Scores (%)")
plt.title("Comparison of F1 Scores by Model and Category")
plt.xticks(x + bar_width * 1.5, categories, rotation=45)
plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
plt.tight_layout()

# Show the plot
plt.show()

