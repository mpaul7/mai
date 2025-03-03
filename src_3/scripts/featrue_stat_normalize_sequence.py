import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from pathlib import Path
"""
This code is normalise the stat featurs and also create a sequence features of all the features, 
The new column name based on statistical featuers is "stat_features"
"""
root_dir = '/home/mpaul/projects/mpaul/mai/end-to-end-dl-pipeline/artifacts/data_mar02'
data_transformation_dir = Path(root_dir, 'data_transformation')
data_preparation_dir = Path(root_dir, 'data_preparation')
data_split_dir = Path(root_dir, 'data_split')

data_sources_solana = ['2020a_Wireline_Ethernet', 
                    '2020c_Mobile_Wifi',
                    '2021a_Wireline_Ethernet', 
                    '2021c_Mobile_LTE',
                    '2022a_Wireline_Ethernet', 
                    '2023a_Wireline_Ethernet',
                    '2023c_Mobile_LTE', 
                    '2023e_MacOS_Wifi', 
                    '2024ag_Wireline_Ethernet',
                    '2024a_Wireline_Ethernet', 
                    '2024cg_Mobile_LTE', 
                    '2024c_Mobile_LTE',
                    '2024e_MacOS_Wifi'
                    ]

data_sources_homeoffice = ['Homeoffice2024ag_Wireline_Ethernet',
                    'Homeoffice2024a_Wireline_Ethernet', 
                    'Homeoffice2024c_Mobile_LTE',
                    'Homeoffice2024e_MacOS_WiFi', 
                    'Homeoffice2025cg_Mobile_LTE'
                    ]

data_sources_solanatest = ['Test2023a_Wireline_Ethernet', 
                    'Test2023c_Mobile_LTE',
                    'Test2023e_MacOS_Wifi', 
                    'Test2024ag_Wireline_Ethernet',
                    'Test2024a_Wireline_Ethernet', 
                    'Test2024cg_Mobile_LTE',
                    'Test2024c_Mobile_LTE', 
                    'Test2024e_MacOS_Wifi'
                    ]

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

stat_features_twc2 = ['pkt_fwd_count', 
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

stat_features_tr = [
    'duration_fwd',
'duration_bwd',
'dsMinPl_fwd',
'dsMinPl_bwd',
'dsMaxPl_fwd',
'dsMaxPl_bwd',
'dsMeanPl_fwd',
'dsMeanPl_bwd',
'dsLowQuartilePl_fwd',
'dsLowQuartilePl_bwd',
'dsMedianPl_fwd',
'dsMedianPl_bwd',
'dsUppQuartilePl_fwd',
'dsUppQuartilePl_bwd',
'dsIqdPl_fwd',
'dsIqdPl_bwd',
'dsModePl_fwd',
'dsModePl_bwd',
'dsRangePl_fwd',
'dsRangePl_bwd',
'dsStdPl_fwd',
'dsStdPl_bwd',
'dsRobStdPl_fwd',
'dsRobStdPl_bwd',
'dsSkewPl_fwd',
'dsSkewPl_bwd',
'dsExcPl_fwd',
'dsExcPl_bwd',
'dsMinIat_fwd',
'dsMinIat_bwd',
'dsMaxIat_fwd',
'dsMaxIat_bwd',
'dsMeanIat_fwd',
'dsMeanIat_bwd',
'dsLowQuartileIat_fwd',
'dsLowQuartileIat_bwd',
'dsMedianIat_fwd',
'dsMedianIat_bwd',
'dsUppQuartileIat_fwd',
'dsUppQuartileIat_bwd',
'dsIqdIat_fwd',
'dsIqdIat_bwd',
'dsModeIat_fwd',
'dsModeIat_bwd',
'dsRangeIat_fwd',
'dsRangeIat_bwd',
'dsStdIat_fwd',
'dsStdIat_bwd',
'dsRobStdIat_fwd',
'dsRobStdIat_bwd',
'dsSkewIat_fwd',
'dsSkewIat_bwd',
'dsExcIat_fwd',
'dsExcIat_bwd',
'PyldEntropy_fwd',
'PyldEntropy_bwd',
'PyldChRatio_fwd',
'PyldChRatio_bwd',
'PyldBinRatio_fwd',
'PyldBinRatio_bwd',
'pktsSnt',
'pktsRcvd',
'l7BytesSnt',
'l7BytesRcvd',
'minL7PktSz_fwd',
'minL7PktSz_bwd',
'maxL7PktSz_fwd',
'maxL7PktSz_bwd',
'avgL7PktSz_fwd',
'avgL7PktSz_bwd',
'stdL7PktSz_fwd',
'stdL7PktSz_bwd',
'minIAT_fwd',
'minIAT_bwd',
'maxIAT_fwd',
'maxIAT_bwd',
'avgIAT_fwd',
'avgIAT_bwd',
'stdIAT_fwd',
'stdIAT_bwd',
'pktps_fwd',
'pktps_bwd',
'bytps_fwd',
'bytps_bwd',
'pktAsm_fwd',
'pktAsm_bwd',
'bytAsm_fwd',
'bytAsm_bwd',
# ====================  
# 'dnsStat'  
# 'dnsAAAqF'  
# 'dnsQname'  
# 'dnsAname'  
# 'dnsAPname'  
# 'dns4Aaddress'  
# 'dnsHdrOPField_fwd'  
# 'dnsHdrOPField_bwd'  
# ====================  
# 'tcpFStat_fwd',
# 'tcpFStat_bwd',
# 'ipMindIPID_fwd',
# 'ipMindIPID_bwd',
# 'ipMaxdIPID_fwd',
# 'ipMaxdIPID_bwd',
# 'ipMinTTL_fwd',
# 'ipMinTTL_bwd',
# 'ipMaxTTL_fwd',
# 'ipMaxTTL_bwd',
# 'ipTTLChg_fwd',
# 'ipTTLChg_bwd',
# 'ipToS_fwd',
# 'ipToS_bwd',
# 'ipFlags',
# 'ipOptCnt',
# ==================  
# 'tcpISeqN_fwd',
# 'tcpISeqN_bwd',
# 'tcpPSeqCnt_fwd',
# 'tcpPSeqCnt_bwd',
# 'tcpSeqSntBytes_fwd',
# 'tcpSeqSntBytes_bwd',
# 'tcpSeqFaultCnt_fwd',
# 'tcpSeqFaultCnt_bwd',
# 'tcpPAckCnt_fwd',
# 'tcpPAckCnt_bwd',
# 'tcpFlwLssAckRcvdBytes_fwd',
# 'tcpFlwLssAckRcvdBytes_bwd',
# 'tcpAckFaultCnt_fwd',
# 'tcpAckFaultCnt_bwd',
# 'tcpBFlgtMx_fwd',
# 'tcpBFlgtMx_bwd',
# 'tcpInitWinSz_fwd',
# 'tcpInitWinSz_bwd',
# 'tcpAvgWinSz_fwd',
# 'tcpAvgWinSz_bwd',
# 'tcpMinWinSz_fwd',
# 'tcpMinWinSz_bwd',
# 'tcpMaxWinSz_fwd',
# 'tcpMaxWinSz_bwd',
# 'tcpWinSzDwnCnt_fwd',
# 'tcpWinSzDwnCnt_bwd',
# 'tcpWinSzUpCnt_fwd',
# 'tcpWinSzUpCnt_bwd',
# 'tcpWinSzChgDirCnt_fwd',
# 'tcpWinSzChgDirCnt_bwd',
# 'tcpWinSzThRt_fwd',
# 'tcpWinSzThRt_bwd',
# 'tcpWS_fwd',
# 'tcpWS_bwd',
# 'tcpFlags_fwd',
# 'tcpFlags_bwd',
# 'tcpAnomaly_fwd',
# 'tcpAnomaly_bwd',
# ====================  
# 'sslStat_fwd',
# 'sslStat_bwd',
# 'sslProto_fwd',
# 'sslProto_bwd',
# 'sslFlags_fwd',
# 'sslFlags_bwd',
# 'sslRecVer_fwd',
# 'sslRecVer_bwd',
# 'sslNumRecVer_fwd',
# 'sslNumRecVer_bwd',
# 'sslNumExt_fwd',
# 'sslNumExt_bwd',
# 'sslNumHandVer_fwd',
# 'sslNumHandVer_bwd',
# 'sslNumSuppVer_fwd',
# 'sslNumSuppVer_bwd',
# 'sslNumSigAlg_fwd',
# 'sslNumSigAlg_bwd',
# 'sslSigAlg_fwd',
# 'sslSigAlg_bwd',
# 'sslNumECPt_fwd',
# 'sslNumECPt_bwd',
# 'sslECPt_fwd',
# 'sslECPt_bwd',
# 'sslNumECFormats_fwd',
# 'sslNumECFormats_bwd',
# 'sslECFormats_fwd',
# 'sslECFormats_bwd'
]

stat_features_twc_test = ["pl_len_fwd_min", "pl_len_fwd_mean", "pl_len_fwd_max", 
"pl_len_fwd_stdev", "pl_len_bwd_min", "pl_len_bwd_mean", 
"pl_len_bwd_max", "pl_len_bwd_stdev", "pkt_len_fwd_min", 
"pkt_len_fwd_mean", "pkt_len_fwd_max", "pkt_len_fwd_stdev", 
"pkt_len_bwd_min", "pkt_len_bwd_mean", "pkt_len_bwd_max", 
"pkt_len_bwd_stdev"]  


# df = pd.read_orc("/home/mpaul/projects/mpaul/mai/data/solana_data_twc_extract/solanatrain-homeoffice_combined.orc")
# df = pd.read_csv('/home/mpaul/projects/mpaul/experiments/Solana_data/solana_2019_2024_data_twc/Solana_twc_extracted_data_beta_all_sources.csv')
df = pd.read_parquet('/home/mpaul/projects/mpaul/mai/end-to-end-dl-pipeline/data_source/tr_ext_Solana_alldata_mapped_100per.parquet')
df = df[stat_features_tr + ['refined_app_label', 'data_source'] ]
print(df.shape, 1)
# df.dropna(subset=stat_features_tr + ['refined_app_label'])
df.dropna()
print(df.shape, 2)

# Step 3: Initialize StandardScaler
scaler = StandardScaler()
print("start fitting")
df[stat_features_tr] = scaler.fit_transform(df[stat_features_tr])
print("start transforming")
df['stat_features'] = df[stat_features_tr].values.tolist()
df.rename(columns={'refined_app_label': 'label'}, inplace=True)

df = df[['stat_features', 'label', 'data_source']].copy()
# Create datasets based on data_source values
d1 = df[df['data_source'].isin(data_sources_solana)].copy()
d2 = df[df['data_source'].isin(data_sources_homeoffice)].copy()
d3 = df[df['data_source'].isin(data_sources_solanatest)].copy()

# Save the datasets to parquet files
d1.to_parquet('/home/mpaul/projects/mpaul/mai/end-to-end-dl-pipeline/artifacts/data_mar02/data_preparation/data_sources_solana.parquet')
d2.to_parquet('/home/mpaul/projects/mpaul/mai/end-to-end-dl-pipeline/artifacts/data_mar02/data_preparation/data_sources_homeoffice.parquet')
d3.to_parquet('/home/mpaul/projects/mpaul/mai/end-to-end-dl-pipeline/artifacts/data_mar02/data_preparation/data_sources_solanatest.parquet')

# Split the data into train and test sets while maintaining class distribution
print("start splitting")
from sklearn.model_selection import train_test_split

train_df, test_df = train_test_split(df, 
                                    test_size=0.2, 
                                    random_state=42,
                                    # stratify=df['refined_app_label']
                                    )
train_df.to_parquet('/home/mpaul/projects/mpaul/mai/end-to-end-dl-pipeline/artifacts/data_mar02/data_split/data_sources_solana_train.parquet')
test_df.to_parquet('/home/mpaul/projects/mpaul/mai/end-to-end-dl-pipeline/artifacts/data_mar02/data_split/data_sources_solana_test.parquet')    

print(f"Training set size: {len(train_df)}")
print(f"Test set size: {len(test_df)}")
