#!/bin/bash
# 
# sleep 18000
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

# Original non-normalized files, contains stat features
train_file="/home/mpaul/projects/mpaul/mai/data/solana_2023c_data_nfs_extract/final_dataset/train_dalhousie_nims_7app_nfs_normalized.parquet"
# test_file="/home/mpaul/projects/mpaul/mai/data/solana_2023c_data_nfs_extract/final_dataset/solana_2023c_7apps_norm_v5.parquet"

# train_file="/home/mpaul/projects/mpaul/mai/data/standard_scaler/final_train_test/train_dalhousie_nims_7app_nfs_normalized.parquet"
# train_file="/home/mpaul/projects/mpaul/mai/data/solana_2023c_data_nfs_extract/final_dataset/train_dalhousie_nims_7app_nfs_normalized_augmented_20per.parquet"
# train_file='/home/mpaul/projects/mpaul/mai/data/solana_2023c_data_nfs_extract/final_dataset/train_dalhousie_nims_7app_nfs_normalized_augmented_20x2per.parquet'
# train_file='/home/mpaul/projects/mpaul/mai/data/solana_2023c_data_nfs_extract/final_dataset/train_dalhousie_nims_7app_nfs_normalized_augmented_20x4per.parquet'
test_file="/home/mpaul/projects/mpaul/mai/data/solana_2023c_data_nfs_extract/final_dataset/solana_2023c_7apps_norm_v6_80per.parquet"

clf_type='dl'


# test_model="/home/mpaul/projects/mpaul/mai/mlruns/808894472966913361/cb3e9aee6fd24d4e83704919f15e7972/artifacts/lstm_100_0.0007_20250127161138.h5"

#l1_0.0001
# test_model="/home/mpaul/projects/mpaul/mai/mlruns/942854525086969943/bef8aa1426224cebbc0906cfb4d4343c/artifacts/cnn_120_0.001_20250130201225.h5"

#l2_0.0001
# test_model="/home/mpaul/projects/mpaul/mai/mlruns/942854525086969943/2a3a484b425945ac8fd456c169334098/artifacts/cnn_120_0.001_20250131011502.h5"

#lr_0.001
# test_model="/home/mpaul/projects/mpaul/mai/mlruns/482621291805992568/4341cadebe9b4ee7a3c5ebb23cb5ed60/artifacts/cnn_120_0.001_20250130155125.h5"

#lr_0.01
# test_model="/home/mpaul/projects/mpaul/mai/mlruns/482621291805992568/795c2c393880417384c7743898c547ef/artifacts/cnn_120_0.001_20250130155125.h5"

#dr_0.1
# test_model="/home/mpaul/projects/mpaul/mai/mlruns/570699845286275433/5d7688dbc3cf412f9de9b24ff4cb016f/artifacts/cnn_120_0.001_20250130114049.h5"

#dr_0.2
# test_model="/home/mpaul/projects/mpaul/mai/mlruns/570699845286275433/faa6938b3a4e41659473c3880cb6deda/artifacts/cnn_120_0.001_20250130114049.h5"

#dr_0.3
# test_model="/home/mpaul/projects/mpaul/mai/mlruns/570699845286275433/8d7c07a8abd64eb58242944825f1a1c9/artifacts/cnn_120_0.001_20250130114049.h5"

# fixed 
 test_model='/home/mpaul/projects/mpaul/mai/mlruns/345364263037389871/ad5a97cbc89d49ee804fa77fdda36dc9/artifacts/cnn_120_0.001_20250131083629.h5'

dl_config_file="/home/mpaul/projects/mpaul/mai/configs/dl/dl_models_y2.json"
output_file="/home/mpaul/projects/mpaul/mai/results/results_jan28/learning_rate/mlp_lstm_cnn_20241203190916_test_results.csv"

# python3 src_5_overfitting_exp/cli.py ${clf_type} train ${train_file} ${dl_config_file}
python3 src_5_overfitting_exp/cli.py ${clf_type} test ${test_model} ${dl_config_file} ${test_file} 
