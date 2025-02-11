import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

"""
This code is normalise the stat featurs and also create a sequence features of all the features, 
The new column name based on statistical featuers is "stat_features"
"""

stat_features_twc = ['pkt_fwd_count', 
    'pl_fwd_count',
    'pl_len_fwd_mean', 
    'pl_len_fwd_stdev',
    'pl_len_fwd_total', 
    'pl_len_fwd_min', 
    'pl_len_fwd_max',
    'pkt_len_fwd_mean', 
    'pkt_len_fwd_stdev', 
    'pkt_len_fwd_total',
    'pkt_len_fwd_min', 
    'pkt_len_fwd_max', 
    'iat_fwd_mean', 
    'iat_fwd_stdev',
    'iat_fwd_total', 
    'iat_fwd_min', 
    'iat_fwd_max', 
    'pkt_bwd_count',
    'pl_bwd_count', 
    'last_timestamp_bwd', 
    'pl_len_bwd_mean',
    'pl_len_bwd_stdev', 
    'pl_len_bwd_total', 
    'pl_len_bwd_min',
    'pl_len_bwd_max', 
    'pkt_len_bwd_mean', 
    'pkt_len_bwd_stdev',
    'pkt_len_bwd_total', 
    'pkt_len_bwd_min', 
    'pkt_len_bwd_max',
    'iat_bwd_mean', 
    'iat_bwd_stdev', 
    'iat_bwd_total', 
    'iat_bwd_min',
    'iat_bwd_max'
]

# df = pd.read_orc("/home/mpaul/projects/mpaul/mai/data/solana_data_twc_extract/solanatrain-homeoffice_combined.orc")
df = pd.read_csv('/home/mpaul/projects/mpaul/experiments/Solana_data/solana_2019_2024_data_twc/Solana_twc_extracted_data_beta_all_sources.csv')
# Remove rows where refined_label_app is None
print(df.shape, 1)
df = df.dropna(subset=['refined_app_label'])
print(df.shape, 2)

# Step 2: Select columns to standardize (example: numeric columns)
stat_features_nfs = [
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

stat_features_twc = ['pkt_fwd_count', 
    'pl_fwd_count',
    'pl_len_fwd_mean', 
    'pl_len_fwd_stdev',
    'pl_len_fwd_total', 
    'pl_len_fwd_min', 
    'pl_len_fwd_max',
    'pkt_len_fwd_mean', 
    'pkt_len_fwd_stdev', 
    'pkt_len_fwd_total',
    'pkt_len_fwd_min', 
    'pkt_len_fwd_max', 
    'iat_fwd_mean', 
    'iat_fwd_stdev',
    'iat_fwd_total', 
    'iat_fwd_min', 
    'iat_fwd_max', 
    'pkt_bwd_count',
    'pl_bwd_count', 
    'last_timestamp_bwd', 
    'pl_len_bwd_mean',
    'pl_len_bwd_stdev', 
    'pl_len_bwd_total', 
    'pl_len_bwd_min',
    'pl_len_bwd_max', 
    'pkt_len_bwd_mean', 
    'pkt_len_bwd_stdev',
    'pkt_len_bwd_total', 
    'pkt_len_bwd_min', 
    'pkt_len_bwd_max',
    'iat_bwd_mean', 
    'iat_bwd_stdev', 
    'iat_bwd_total', 
    'iat_bwd_min',
    'iat_bwd_max'
]

"""
    A subset of the features are used for the test dataset.
"""
stat_features_twc_test = ["pl_len_fwd_min", "pl_len_fwd_mean", "pl_len_fwd_max", 
"pl_len_fwd_stdev", "pl_len_bwd_min", "pl_len_bwd_mean", 
"pl_len_bwd_max", "pl_len_bwd_stdev", "pkt_len_fwd_min", 
"pkt_len_fwd_mean", "pkt_len_fwd_max", "pkt_len_fwd_stdev", 
"pkt_len_bwd_min", "pkt_len_bwd_mean", "pkt_len_bwd_max", 
"pkt_len_bwd_stdev"]  

# Step 3: Initialize StandardScaler
scaler = StandardScaler()

# Step 4: Fit and transform the selected columns
print("start fitting")
df[stat_features_twc] = scaler.fit_transform(df[stat_features_twc])
print("start transforming")
df['stat_features'] = df[stat_features_twc].values.tolist()
# df['stat_features_test'] = df[stat_features_twc_test].values.tolist()

# df.to_csv("/home/mpaul/projects/mpaul/mai/data/solana_data_twc_extract/solanatrain-homeoffice_combined_normalized_stat_features_100percent.csv")
# df.to_parquet("/home/mpaul/projects/mpaul/mai/data/solana_data_twc_extract/solanatrain-homeoffice_combined_normalized_stat_features_100percent.parquet")
# df.to_csv('/home/mpaul/projects/mpaul/experiments/Solana_data/solana_2019-2024_beta2/solana_beta_2/solana_nfs_ext_twc_mapped_labels_100percent.csv')

# Split the data into train and test sets while maintaining class distribution
print("start splitting")
from sklearn.model_selection import train_test_split

train_df, test_df = train_test_split(df, 
                                    test_size=0.2, 
                                    random_state=42,
                                    # stratify=df['refined_app_label']
                                    )

# Save train and test splits
# train_df.to_csv("/home/mpaul/projects/mpaul/mai/data/solana_data_twc_extract/solanatrain-homeoffice_combined_normalized_stat_features_train_80percent.csv")
# train_df.to_parquet("/home/mpaul/projects/mpaul/mai/data/solana_data_twc_extract/solanatrain-homeoffice_combined_normalized_stat_features_train_80percent.parquet")

# test_df.to_csv("/home/mpaul/projects/mpaul/mai/data/solana_data_twc_extract/solanatrain-homeoffice_combined_normalized_stat_features_test_20percent.csv") 
# test_df.to_parquet("/home/mpaul/projects/mpaul/mai/data/solana_data_twc_extract/solanatrain-homeoffice_combined_normalized_stat_features_test_20percent.parquet")

train_df.to_csv('/home/mpaul/projects/mpaul/experiments/Solana_data/solana_2019_2024_data_twc/processed_data/Solana_twc_extracted_data_beta_all_sources_train_80percent.csv')
test_df.to_csv('/home/mpaul/projects/mpaul/experiments/Solana_data/solana_2019_2024_data_twc/processed_data/Solana_twc_extracted_data_beta_all_sources_test_20percent.csv')    

train_df.to_parquet('/home/mpaul/projects/mpaul/experiments/Solana_data/solana_2019_2024_data_twc/processed_data/Solana_twc_extracted_data_beta_all_sources_train_80percent.parquet')
test_df.to_parquet('/home/mpaul/projects/mpaul/experiments/Solana_data/solana_2019_2024_data_twc/processed_data/Solana_twc_extracted_data_beta_all_sources_test_20percent.parquet')    

print(f"Training set size: {len(train_df)}")
print(f"Test set size: {len(test_df)}")
