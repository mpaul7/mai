import pandas as pd
import numpy as np

# Read the parquet file


# Create stat_features column from all the specified columns
feature_columns = [
    'bidirectional_duration_ms', 'bidirectional_packets',
    'bidirectional_bytes', 'src2dst_duration_ms', 'src2dst_packets',
    # 'src2dst_bytes', 'dst2src_duration_ms', 'dst2src_packets',
    # 'dst2src_bytes', 'bidirectional_min_ps', 'bidirectional_mean_ps',
    # 'bidirectional_stddev_ps', 'bidirectional_max_ps', 'src2dst_min_ps',
    # 'src2dst_mean_ps', 'src2dst_stddev_ps', 'src2dst_max_ps',
    # 'dst2src_min_ps', 'dst2src_mean_ps', 'dst2src_stddev_ps',
    # 'dst2src_max_ps', 'bidirectional_min_piat_ms',
    # 'bidirectional_mean_piat_ms', 'bidirectional_stddev_piat_ms',
    # 'bidirectional_max_piat_ms', 'src2dst_min_piat_ms',
    # 'src2dst_mean_piat_ms', 'src2dst_stddev_piat_ms', 'src2dst_max_piat_ms',
    # 'dst2src_min_piat_ms', 'dst2src_mean_piat_ms', 'dst2src_stddev_piat_ms',
    # 'dst2src_max_piat_ms', 'bidirectional_syn_packets',
    # 'bidirectional_cwr_packets', 'bidirectional_ece_packets',
    # 'bidirectional_urg_packets', 'bidirectional_ack_packets',
    # 'bidirectional_psh_packets', 'bidirectional_rst_packets',
    # 'bidirectional_fin_packets', 'src2dst_syn_packets',
    # 'src2dst_cwr_packets', 'src2dst_ece_packets', 'src2dst_urg_packets',
    # 'src2dst_ack_packets', 'src2dst_psh_packets', 'src2dst_rst_packets',
    # 'src2dst_fin_packets', 'dst2src_syn_packets', 'dst2src_cwr_packets',
    # 'dst2src_ece_packets', 'dst2src_urg_packets', 'dst2src_ack_packets',
    # 'dst2src_psh_packets', 'dst2src_rst_packets', 'dst2src_fin_packets'
]
df = pd.read_parquet('/home/mpaul/projects/mpaul/mai/data/test_solana_7apps_nfs_v3.parquet', )

df = df[feature_columns]
# Convert rows to lists and store in stat_features column
df['stat_features'] = df[feature_columns].values.tolist()

# Print a few examples to verify
print(df['stat_features'].head())
df.head(20).to_csv('/home/mpaul/projects/mpaul/mai/data/test_solana_7apps_nfs_v4.csv')