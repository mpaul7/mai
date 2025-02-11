#!/bin/bash
# sleep 10800

# train_file="/home/mpaul/projects/mpaul/mai2/data/solana_data_twc_extract/solanatrain-homeoffice_combined_normalized_stat_features_train_80percent.parquet"
# train_file="/home/mpaul/projects/mpaul/mai/data/solana_2023c_data_nfs_extract/final_dataset/train_dalhousie_nims_7app_nfs_normalized.parquet"
# test_file="/home/mpaul/projects/mpaul/mai2/data/solana_data_twc_extract/solanatrain-homeoffice_combined_normalized_stat_features_test_20percent.parquet"

# train_file="/home/mpaul/projects/mpaul/experiments/Solana_data/solana_2019-2024_beta2/solana_beta_2/processed_data/solana_nfs_ext_twc_mapped_labels_train_80percent.parquet"
test_file="/home/mpaul/projects/mpaul/experiments/Solana_data/solana_2019-2024_beta2/solana_beta_2/processed_data/solana_nfs_ext_twc_mapped_labels_test_20percent.parquet"
train_file="/home/mpaul/projects/mpaul/experiments/Solana_data/solana_2019_2024_data_twc/processed_data/Solana_twc_extracted_data_beta_all_sources_train_80percent.parquet"
# test_file="/home/mpaul/projects/mpaul/experiments/Solana_data/solana_2019_2024_data_twc/processed_data/Solana_twc_extracted_data_beta_all_sources_test_20percent.parquet"

clf_type='dl'

test_model='/home/mpaul/projects/mpaul/mai/mlruns/345364263037389871/ad5a97cbc89d49ee804fa77fdda36dc9/artifacts/cnn_120_0.001_20250131083629.h5'


dl_config_file="/home/mpaul/projects/mpaul/mai2/configs/dl/dl_models_y2.json"
output_file="/home/mpaul/projects/mpaul/mai/results/results_jan28/learning_rate/mlp_lstm_cnn_20241203190916_test_results.csv"

python3 src_3/cli.py ${clf_type} train ${train_file} ${dl_config_file}
# python3 src_3/dl_models.py ${clf_type} test ${test_model} ${dl_config_file} ${test_file} 
