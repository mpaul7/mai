import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

"""
This code is normalise the stat featurs and also create a sequence features of all the features, 
The new column name based on statistical featuers is "stat_features"
"""

df = pd.read_orc("/home/mpaul/projects/mpaul/mai/data/solana_data_twc_extract/solanatrain-homeoffice_combined.orc")


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
    A subset of the features are used for this test 
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
df[stat_features_twc] = scaler.fit_transform(df[stat_features_twc])
df['stat_features'] = df[stat_features_twc].values.tolist()
df['stat_features_test'] = df[stat_features_twc_test].values.tolist()
"""
    Note: df['stat_features_test'] is used as an array of statistical features for CNN model. 
"""

