#!/bin/bash

train_file="/home/mpaul/projects/mpaul/mai/data/solana_2023c_data_nfs_extract/final_dataset/train_dalhousie_nims_7app_nfs_normalized_augmented_20per.parquet"
test_file="/home/mpaul/projects/mpaul/mai/data/solana_2023c_data_nfs_extract/final_dataset/solana_2023c_7apps_norm_v6_80per.parquet"

# =============== DL Models ===================== 
clf_type='dl'
test_model='/home/mpaul/projects/mpaul/mai/models/models_jan13/mlp_120_0.0001_20250113155400.h5'
dl_config_file="/home/mpaul/projects/mpaul/mai/configs/dl/dl_models_enta.json"
output_file="/home/mpaul/projects/mpaul/mai/results/mlp_lstm_cnn_20241203190916_test_results.csv"

python3 src_4_enta/cli_enta.py ${clf_type} train ${train_file} ${dl_config_file}
# python3 /home/mpaul/projects/mpaul/mai/src_3/cli.py ${clf_type} test ${test_model} ${dl_config_file} ${test_file}
