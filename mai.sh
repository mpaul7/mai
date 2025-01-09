#!/bin/bash

# train_file="/home/mpaul/projects/mpaul/mai/data/train_cryptocurrncy_70files_2.parquet"
# test_file="/home/mpaul/projects/mpaul/mai/data/test_cryptocurrncy_70files_2.parquet"

# train_file="/home/mpaul/projects/mpaul/mai/data/dns/bucketized_data/dns_30s_train_data.parquet"
# test_file="/home/mpaul/projects/mpaul/mai/data/dns/bucketized_data/dns_30s_test_data.parquet"

# train_file="/home/mpaul/projects/mpaul/mai/data/train_dalhousie_nims_7app_nfs.parquet"

# Original non-normalized files, contains stat features
# train_file="/home/mpaul/projects/mpaul/mai/data/train_dalhousie_nims_7app_nfs_stat_features_v2.parquet"
# test_file="/home/mpaul/projects/mpaul/mai/data/test_dalhousie_nims_7app_nfs_v2.parquet"

# Normalized files, contains stat features
# train_file="/home/mpaul/projects/mpaul/mai/data/train_dalhousie_nims_7app_nfs_stat_features_v2_normalized.parquet"
# test_file="/home/mpaul/projects/mpaul/mai/data/test_dalhousie_nims_7app_nfs_v2_normalized.parquet"

train_file="/home/mpaul/projects/mpaul/mai/data/standard_scaler/final_train_test/train_dalhousie_nims_7app_nfs_normalized.parquet"
# test_file="/home/mpaul/projects/mpaul/mai/data/standard_scaler/final_train_test/test_dalhousie_nims_7app_nfs_normalized.parquet"
# test_file="/home/mpaul/projects/mpaul/mai/data/standard_scaler/final_train_test/test_solana_7app_nfs_normalized.parquet"
test_file="/home/mpaul/projects/mpaul/mai/data/solana_2023c_data_nfs_extract/solana_2023c_7apps_nfs_extraxted_enta_task_normalized.parquet"
# test_file="/home/mpaul/projects/mpaul/mai/data/solana_2023c_data_nfs_extract/solana_2023c_7apps_nfs_extraxted_enta_task_non_normalized.parquet"

# =============== DL Models ===================== 
clf_type='dl'


# test_model='/home/mpaul/projects/mpaul/mai/models_jan06/1_mlp_120_20250106215848.h5'
test_model='/home/mpaul/projects/mpaul/mai/models_jan06/2_lstm_120_20250106223118.h5'
# test_model='/home/mpaul/projects/mpaul/mai/models_jan06/3_cnn_60_20250106230825.h5'
# test_model='/home/mpaul/projects/mpaul/mai/models_jan06/4_cnn_120_20250107115237.h5'
# test_model='/home/mpaul/projects/mpaul/mai/models_jan06/5_mlp_lstm_120_20250107111953.h5'
# test_model='/home/mpaul/projects/mpaul/mai/models_jan06/6_mlp_cnn_seq_120_20250107065342.h5'
# test_model='/home/mpaul/projects/mpaul/mai/models_jan06/7_lstm_cnn_sta_60_20250107133200.h5'
# test_model='/home/mpaul/projects/mpaul/mai/models_jan06/8_mlp_lstm_60_20250107183834.h5'
# test_model='/home/mpaul/projects/mpaul/mai/models_jan06/9_mlp_cnn_seq_30_20250107182413.h5'
# test_model='/home/mpaul/projects/mpaul/mai/models_jan06/10_lstm_cnn_sta_60_20250107144323.h5'
# dl_config_file="configs/${clf_type}/${model}_dalhousie_nims_config_v2.json"
dl_config_file="/home/mpaul/projects/mpaul/mai/configs/dl/dl_models.json"
output_file="/home/mpaul/projects/mpaul/mai/results/mlp_lstm_cnn_20241203190916_test_results.csv"

# python3 src_3/cli.py ${clf_type} train ${train_file} ${dl_config_file}
python3 /home/mpaul/projects/mpaul/mai/src_3/cli.py ${clf_type} test ${test_model} ${dl_config_file} ${test_file}

# =============== ML Models ====================
# clf_type='ml'
# bucket_size=30
# model="isolatiopn_forest"
# config="isolationForest_dns"
# ml_config_file="configs/${clf_type}/${config}.json"
# trainded_model_name="${config}_${bucket_size}s"
# python3 src/cli.py ${clf_type} train ${train_file} ${ml_config_file} ${trainded_model_name}
# python3 src/cli.py ${clf_type} test ${trainded_model_name} ${test_file} ${ml_config_file}
