import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import confusion_matrix, classification_report

import numpy as np

# Load the training data
train_data = pd.read_csv('/home/mpaul/projects/mpaul/mai/data/dns/final_data/train_cubro_dns_pcap_00002375_July24-25_20241113164247.csv')

# Separate features and labels
target_labels = ['pl_fwd_count', 'pl_len_fwd_total', 'pl_bwd_count', 'pl_len_bwd_total']
X_train = train_data[target_labels]
y_train = train_data['label']

# Create a pipeline with StandardScaler and IsolationForest
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('iso_forest', IsolationForest(random_state=42))
])

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'iso_forest__contamination': [0.01, 0.05, 0.1],  # Example values
    'iso_forest__n_estimators': [25, 50, 100, 200]  # Example values
}

# Set up KFold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Create GridSearchCV object
grid_search = GridSearchCV(pipeline, param_grid, cv=kf, scoring='f1', n_jobs=-1)

# Fit the model using GridSearchCV
grid_search.fit(X_train, y_train)

# Best parameters found
print("Best parameters found: ", grid_search.best_params_)

# Load the evaluation data
test_data = pd.read_csv('/home/mpaul/projects/mpaul/mai/data/dns/final_data/test_cubro_Solana-DNS-Amplification-3Attacker-100.csv')

# Separate features and labels
X_test = test_data[target_labels]
y_test = test_data['label']

# Predict anomalies in the test data using the best model
y_pred = grid_search.predict(X_test)

# Convert predictions to match the label format
# 1 for 'dns' (normal) and 0 for 'dns_attack' (outlier)
y_pred_labels = np.where(y_pred == -1, 'dns_attack', 'dns')

# Generate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_labels, labels=['dns', 'dns_attack'])

# Print the confusion matrix and classification report
print("Confusion Matrix:")
print(conf_matrix)

print("\nClassification Report:")
print(classification_report(y_test, y_pred_labels, target_names=['dns', 'dns_attack']))

# To get the outliers from the test data
print(y_pred_labels)

outliers = test_data[y_pred_labels == 'dns_attack']

print("\nOutliers in the test data:")
print(outliers)