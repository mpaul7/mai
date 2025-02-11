import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import ast

def string_to_array(s):
    # Clean the string and convert to list of numbers
    # Remove extra spaces and brackets
    s = s.strip('[]').replace('  ', ' ').strip()
    # Convert to list of integers
    return np.array([int(x) for x in s.split() if x])

# Read the CSV file
df = pd.read_csv('/home/mpaul/projects/mpaul/mai/data/standard_scaler/test_data_3.csv', index_col=0)

# Convert string arrays to numpy arrays
df['splt_ps'] = df['splt_ps'].apply(string_to_array)
df['splt_piat'] = df['splt_piat'].apply(string_to_array)

# Create separate scalers for each column
scaler_ps = StandardScaler()
scaler_piat = StandardScaler()

# Stack all arrays from each column to create 2D arrays
ps_stacked = np.vstack(df['splt_ps'].values)
piat_stacked = np.vstack(df['splt_piat'].values)

# Fit and transform the data
ps_scaled = scaler_ps.fit_transform(ps_stacked)
piat_scaled = scaler_piat.fit_transform(piat_stacked)

# Create DataFrames with scaling parameters
# ps_params = pd.DataFrame(
#     {'mean': scaler_ps.mean_,
#      'std': scaler_ps.scale_},
#     index=[f'position_{i+1}' for i in range(len(scaler_ps.mean_))]
# )

# piat_params = pd.DataFrame(
#     {'mean': scaler_piat.mean_,
#      'std': scaler_piat.scale_},
#     index=[f'position_{i+1}' for i in range(len(scaler_piat.mean_))]
# )

# Print scaling parameters
# print("Scaling parameters for splt_ps:")
# print(ps_params)
# print("\nScaling parameters for splt_piat:")
# print(piat_params)

# If you want to save the scaled data back to arrays
df_scaled = pd.DataFrame({
    'splt_ps': [row for row in ps_scaled],
    'splt_piat': [row for row in piat_scaled]
}, index=df.index)

# Save to CSV if needed
df_scaled.to_csv('/home/mpaul/projects/mpaul/mai/data/standard_scaler/test_data_3_scaledList.csv')

# Print sample of original vs scaled data
print("\nOriginal first array in splt_ps:")
print(df['splt_ps'].iloc[0])
print("\nScaled first array in splt_ps:")
print(ps_scaled[0])