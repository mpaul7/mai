import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Read the CSV file (assuming your data is saved as 'data.csv')
df = pd.read_csv('/home/mpaul/projects/mpaul/mai/data/standard_scaler/test_data_4.csv', index_col=0)  # index_col=0 because first column is index

# Initialize a StandardScaler
scaler = StandardScaler()

# Create a copy of the dataframe and standardize all columns
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)

# If you want to see the mean and standard deviation for each column
scaling_params = pd.DataFrame(
    {'mean': scaler.mean_,
     'std': scaler.scale_},
    index=df.columns
)

print("Original first few rows:")
print(df.head())
print("\nScaled first few rows:")
print(df_scaled.head())
print("\nScaling parameters:")
print(scaling_params)

# If you need to save the scaled data
df_scaled.to_csv('/home/mpaul/projects/mpaul/mai/data/standard_scaler/test_data_4_scaler_columnwise.csv')