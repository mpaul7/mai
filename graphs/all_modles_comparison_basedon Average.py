import matplotlib.pyplot as plt

# Model names and their corresponding mean accuracies
models = ['CNN', 'LSTM', 'MLP', 'RF', 'XGBoost']
mean_accuracies = [93.38, 84.78, 68.66, 93.14, 93.57]

# Create a bar graph
plt.figure(figsize=(10, 6))
plt.bar(models, mean_accuracies, color=['salmon', 'orange', 'lightgreen', 'grey', 'skyblue'])

# Adding title and labels
plt.title('Comparison of ML and DL models based on average accuracy')
plt.xlabel('Models')
plt.ylabel('Mean Accuracy (%)')
plt.ylim(0, 100)  # Set y-axis limit for better visualization

# Adding the data labels on top of the bars
for i, v in enumerate(mean_accuracies):
    plt.text(i, v + 1, f"{v:.2f}", ha='center', fontweight='bold')

# Show the plot
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

