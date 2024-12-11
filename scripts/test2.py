import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def generate_synthetic_dns_data(n_normal=1000, n_anomalies=50):
    normal_data = np.random.normal(loc=0, scale=1, size=(n_normal, 5))  # Normal DNS activity
    anomalies = np.random.normal(loc=4, scale=0.5, size=(n_anomalies, 5))  # Simulated attacks
    data = np.vstack((normal_data, anomalies))
    labels = np.hstack((np.zeros(n_normal), np.ones(n_anomalies)))  # 0 for normal, 1 for attack
    return pd.DataFrame(data), labels

X, y = generate_synthetic_dns_data()

print(X)
print(y)