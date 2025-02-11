import tensorflow as tf
from lstm_overfitting_experiment import LSTMExperiment
import utils_dl

train_file='/home/mpaul/projects/mpaul/mai/data/solana_2023c_data_nfs_extract/final_dataset/train_dalhousie_nims_7app_nfs_normalized_augmented_20x4per.parquet'
test_file="/home/mpaul/projects/mpaul/mai/data/solana_2023c_data_nfs_extract/final_dataset/solana_2023c_7apps_norm_v6_80per.parquet"

def prepare_data():
    """
    Prepare your dataset here
    Returns:
        train_dataset: TF dataset for training
        val_dataset: TF dataset for validation
        input_shape: Shape of input data
        num_classes: Number of output classes
    """
    # Replace this with your actual data preparation code
    # This is just an example
    train_dataset = utils_dl.create_train_test_dataset_tf(
        data_file=train_file,
        params=params,
        train=True,
        evaluation=False
    )
    
    val_dataset = utils_dl.create_train_test_dataset_tf(
        data_file=test_file,
        params=params,
        train=False,
        evaluation=True
    )
    
    return train_dataset, val_dataset, input_shape, num_classes

def main():
    # Prepare data
    train_dataset, val_dataset, input_shape, num_classes = prepare_data()
    
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