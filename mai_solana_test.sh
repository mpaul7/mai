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

# test_model='/home/mpaul/projects/mpaul/mai2/mlruns/817613632967417826/95a97a29d0fb4e67a9fc42c460848d59/artifacts/mlp_120_0.01_20250205092534.h5' # mlp1
# test_model='/home/mpaul/projects/mpaul/mai2/mlruns/817613632967417826/121ae413c8504da8a7df5cdf6127f0c9/artifacts/mlp_120_0.01_20250205110513.h5' # mlp2
# test_model='/home/mpaul/projects/mpaul/mai2/mlruns/817613632967417826/4598c271ace94822b07118450e7d3b58/artifacts/mlp_120_0.01_20250205094214.h5' # mlp3 
# test_model='/home/mpaul/projects/mpaul/mai2/mlruns/817613632967417826/2d6faec9e0e648248334d0006ff1dafa/artifacts/mlp_120_0.01_20250205111411.h5' # mlp4 
# test_model='/home/mpaul/projects/mpaul/mai2/mlruns/817613632967417826/6866dad8cf3a48f6974078591dd44041/artifacts/mlp_120_0.01_20250205085648.h5' # mlp5 
# test_model='/home/mpaul/projects/mpaul/mai2/mlruns/817613632967417826/1b587d02c6e246b7a2394c848db67cb9/artifacts/mlp_120_0.01_20250205105647.h5' # mlp6
# test_model='/home/mpaul/projects/mpaul/mai2/mlruns/817613632967417826/8afbdf6674784bf2af34e465b1f7be78/artifacts/mlp_120_0.01_20250205081113.h5' # mlp7
# test_model='/home/mpaul/projects/mpaul/mai2/mlruns/817613632967417826/b4031e5fd64240f2ab7d1afad38560bc/artifacts/mlp_120_0.01_20250205104551.h5' # mlp8

# test_model='/home/mpaul/projects/mpaul/mai2/mlruns/817613632967417826/b0e4dacff3f241cc884a2c4269bab433/artifacts/cnn_120_0.01_20250204193438.h5' # cnn1
# test_model='/home/mpaul/projects/mpaul/mai2/mlruns/817613632967417826/0f4f9b6d7b034ccbb63df5c89531868e/artifacts/cnn_120_0.01_20250204210237.h5' # cnn2
# test_model='/home/mpaul/projects/mpaul/mai2/mlruns/817613632967417826/478cc59bc0b542b7b8457bd9ab0b7a4d/artifacts/cnn_120_0.01_20250204182933.h5' # cnn3
# test_model='/home/mpaul/projects/mpaul/mai2/mlruns/817613632967417826/56f9d52303774a40bad9a28f85f4b35c/artifacts/cnn_120_0.01_20250204163238.h5' # cnn4


# test_model="/home/mpaul/projects/mpaul/mai2/mlruns/411826991417427610/e59c014398c5492c8e1133b01c2de9f1/artifacts/mlp_120_0.01_20250209173046.h5"
# test_model="/home/mpaul/projects/mpaul/mai2/mlruns/411826991417427610/7a655581b79d4e02ab9b9603f4126d98/artifacts/mlp_120_0.01_20250209172554.h5"
# test_model="/home/mpaul/projects/mpaul/mai2/mlruns/411826991417427610/66f5c6afb5f94329acdb3bc0a473cded/artifacts/cnn_120_0.01_20250209141440.h5"
# test_model="/home/mpaul/projects/mpaul/mai2/mlruns/411826991417427610/4a911a7d0ed94ae9bc5d46f04b4a6f17/artifacts/cnn_120_0.01_20250208203635.h5"
test_model="/home/mpaul/projects/mpaul/mai2/mlruns/411826991417427610/402acc21c5dc477c934d855deabe4a04/artifacts/mlp_120_0.01_20250208202707.h5"



dl_config_file="/home/mpaul/projects/mpaul/mai2/configs/dl/dl_models_y2_solana.json"
output_file="/home/mpaul/projects/mpaul/mai/results/results_jan28/learning_rate/mlp_lstm_cnn_20241203190916_test_results.csv"

python3 src_5_overfitting_exp/cli.py ${clf_type} train ${train_file} ${dl_config_file}
# python3 src_5_overfitting_exp/cli.py ${clf_type} test ${test_model} ${dl_config_file} ${test_file} 

