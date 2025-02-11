import json
import tensorflow as tf
from lstm_overfitting_experiment import LSTMExperiment
import utils_dl

train_file='/home/mpaul/projects/mpaul/mai/data/solana_2023c_data_nfs_extract/final_dataset/train_dalhousie_nims_7app_nfs_normalized_augmented_20x4per.parquet'
test_file="/home/mpaul/projects/mpaul/mai/data/solana_2023c_data_nfs_extract/final_dataset/solana_2023c_7apps_norm_v6_80per.parquet"

def prepare_data(params, train_file, test_file):
    """
    Prepare your dataset here
    Returns:
        train_dataset: TF dataset for training
        val_dataset: TF dataset for validation
        input_shape: Shape of input data
        num_classes: Number of output classes
    """
   
    train_dataset, val_dataset = utils_dl.create_train_test_dataset_tf(
            data_file=train_file,
            params=params,
            train=True,
            evaluation=False
        )
    
    return train_dataset, val_dataset

def main():
    config_file = '/home/mpaul/projects/mpaul/mai/configs/dl/overfitting_params.json'
    train_file='/home/mpaul/projects/mpaul/mai/data/solana_2023c_data_nfs_extract/final_dataset/train_dalhousie_nims_7app_nfs_normalized_augmented_20x4per.parquet'
    test_file="/home/mpaul/projects/mpaul/mai/data/solana_2023c_data_nfs_extract/final_dataset/solana_2023c_7apps_norm_v6_80per.parquet"
    with open(config_file) as f:
        params = json.load(f)
    
    # Prepare data
    train_dataset, val_dataset = prepare_data(params, train_file, test_file)
    
    # Create experiment
    experiment = LSTMExperiment(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        input_shape=input_shape,
        num_classes=num_classes
    )
    
    # Run all experiments
    experiment.run_all_experiments()

if __name__ == "__main__":
    main()