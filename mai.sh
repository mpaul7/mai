#!/bin/bash

# train_file="/home/mpaul/projects/mpaul/mai/data/train_cryptocurrncy_70files_2.parquet"
# test_file="/home/mpaul/projects/mpaul/mai/data/test_cryptocurrncy_70files_2.parquet"

# train_file="/home/mpaul/projects/mpaul/mai/data/dns/bucketized_data/dns_30s_train_data.parquet"
# test_file="/home/mpaul/projects/mpaul/mai/data/dns/bucketized_data/dns_30s_test_data.parquet"

# train_file="/home/mpaul/projects/mpaul/mai/data/train_dalhousie_nims_7app_nfs.parquet"
train_file="/home/mpaul/projects/mpaul/mai/data/train_dalhousie_nims_7app_nfs_stat_features_v2.parquet"
test_file="/home/mpaul/projects/mpaul/mai/data/test_dalhousie_nims_7app_nfs.parquet"

# =============== DL Models ===================== 
clf_type='dl'
test_model='/home/mpaul/projects/mpaul/mai/models_dec09a/mlp_cnn_120_20241209213209.h5'
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
