import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

# use this file 

def string_to_array(s):
    # Clean the string and convert to list of numbers
    # Remove brackets, spaces, and split by comma
    s = s.strip('[]').replace(' ', '')
    # Convert to list of integers, splitting by comma
    return np.array([int(x) for x in s.split(',') if x])
# Step 1: Read the CSV file
file_path = "your_file.csv"  # Replace with the path to your CSV file
# df = pd.read_parquet("/home/mpaul/projects/enta_dl/enta_workspace/enta_data/train_test_data/final_processed_data/test_v2.parquet")
# df = pd.read_parquet("/home/mpaul/projects/mpaul/mai/data/train_dalhousie_nims_7app_nfs.parquet")
# df = pd.read_parquet("/home/mpaul/projects/mpaul/mai/data/test_dalhousie_nims_7app_nfs.parquet")
df = pd.read_csv("/home/mpaul/projects/mpaul/mai/data/solana_2023c_data_nfs_extract/2023c_Mobile_LTE_nfs_extract_v2.csv")

# Step 2: Select columns to standardize (example: numeric columns)
stat_features = [
    'bidirectional_duration_ms', 'bidirectional_packets',
    'bidirectional_bytes', 'src2dst_duration_ms', 'src2dst_packets',
    'src2dst_bytes', 'dst2src_duration_ms', 'dst2src_packets',
    'dst2src_bytes', 'bidirectional_min_ps', 'bidirectional_mean_ps',
    'bidirectional_stddev_ps', 'bidirectional_max_ps', 'src2dst_min_ps',
    'src2dst_mean_ps', 'src2dst_stddev_ps', 'src2dst_max_ps',
    'dst2src_min_ps', 'dst2src_mean_ps', 'dst2src_stddev_ps',
    'dst2src_max_ps', 'bidirectional_min_piat_ms',
    'bidirectional_mean_piat_ms', 'bidirectional_stddev_piat_ms',
    'bidirectional_max_piat_ms', 'src2dst_min_piat_ms',
    'src2dst_mean_piat_ms', 'src2dst_stddev_piat_ms', 'src2dst_max_piat_ms',
    'dst2src_min_piat_ms', 'dst2src_mean_piat_ms', 'dst2src_stddev_piat_ms',
    'dst2src_max_piat_ms', 'bidirectional_syn_packets',
    'bidirectional_cwr_packets', 'bidirectional_ece_packets',
    'bidirectional_urg_packets', 'bidirectional_ack_packets',
    'bidirectional_psh_packets', 'bidirectional_rst_packets',
    'bidirectional_fin_packets', 'src2dst_syn_packets',
    'src2dst_cwr_packets', 'src2dst_ece_packets', 'src2dst_urg_packets',
    'src2dst_ack_packets', 'src2dst_psh_packets', 'src2dst_rst_packets',
    'src2dst_fin_packets', 'dst2src_syn_packets', 'dst2src_cwr_packets',
    'dst2src_ece_packets', 'dst2src_urg_packets', 'dst2src_ack_packets',
    'dst2src_psh_packets', 'dst2src_rst_packets', 'dst2src_fin_packets'
]  # Replace with actual column names

# Step 3: Initialize StandardScaler
scaler = StandardScaler()

# Step 4: Fit and transform the selected columns
df[stat_features] = scaler.fit_transform(df[stat_features])
df['stat_features'] = df[stat_features].values.tolist()



seq_features = ['splt_piat', 'splt_ps']
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

df_seq_scaled = pd.DataFrame({
    'splt_ps': [row for row in ps_scaled],
    'splt_piat': [row for row in piat_scaled]
}, index=df.index)

print(df.shape)
print(df_seq_scaled.shape)
print(df_seq_scaled)

print(df_seq_scaled['splt_ps'].values.tolist())
# After creating df_seq_scaled, merge it with the main df
df['splt_ps'] = df_seq_scaled['splt_ps'].values
df['splt_piat'] = df_seq_scaled['splt_piat'].values

# print(df.splt_ps)

print(df['splt_piat'].iloc[0])

# # Now df contains both the original and scaled sequential features
# print("Final DataFrame shape:", df.shape)
# print("Columns in final DataFrame:", df.columns.tolist())

# df.to_csv("/home/mpaul/projects/mpaul/mai/data/standard_scaler/test_solana_7app_nfs_normalized.csv")
# df.to_parquet("/home/mpaul/projects/mpaul/mai/data/standard_scaler/final_train_test/test_solana_7app_nfs_normalized.parquet")

# df.to_csv("/home/mpaul/projects/mpaul/mai/data/solana_2023c_data_nfs_extract/2023c_Mobile_LTE_nfs_extract_v2_normalized.csv")
# print(df.label.unique())
# df.to_parquet("/home/mpaul/projects/mpaul/mai/data/solana_2023c_data_nfs_extract/2023c_Mobile_LTE_nfs_extract_V2.csv_normalized.parquet")
# If you want to save the final DataFrame
# df.to_parquet("/home/mpaul/projects/mpaul/mai/data/standard_scaler/train_dalhousie_nims_7app_nfs_StandardScaler_v2.parquet", index=False)
