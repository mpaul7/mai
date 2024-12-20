import os
import pandas as pd
import logging
from dns_attack_model import DNSAttackModel
from data_preparation import DataPreparation
from dns_attack_report import DNSAttackReport
import yaml
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Existing warning suppression
import warnings 
warnings.filterwarnings('ignore') 

# Load config
with open('/home/mpaul/projects/mpaul/mai/anomly_detection/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Replace constants with config values
ATTACK_INTERVAL = config['settings']['attack_interval']
TYPE = config['settings']['type']
BUCKET_SIZE = config['settings']['bucket_size']
RESULT = config['settings']['result_path']

# Replace feature lists with config values
bucket_target_features = config['features']['bucket_target_features']
flat_target_features = config['features']['flat_target_features']
target_labels = config['features']['target_labels']

data = {
    'train_flat': config['data_paths']['train_flat'],
    'test_flat': config['data_paths']['test_flat'],
    'train_bucket': [],
    'test_bucket': [],
    'test_flat_bucket': []
}

def prepare_data():
    data_preparation = DataPreparation(df=None)
    for key, value in data.items():
        if key in ['train_flat', 'test_flat']:
            dfs = []
            for file in value:
                head, tail = os.path.split(file)
                # print(f'Processing file {file}') 
                df = pd.read_csv(file)
                df = df[(df['sport'] == 53) | (df['dport'] == 53)]
                dfs.append(df)
            merged_df = pd.concat(dfs, ignore_index=True)
            if key == 'train_flat':
                data['train_flat'] = merged_df
                data['train_bucket'], _ = data_preparation.bucketize_data(bucket_size=BUCKET_SIZE, df=merged_df)
            elif key == 'test_flat':
                data['test_flat'] = merged_df 
                data['test_bucket'], data['test_flat_bucket'] = data_preparation.bucketize_data(bucket_size=BUCKET_SIZE, df=merged_df)

    print(f'Data description flow based:\n{"=" * 40}')
    print(f'[{data["train_flat"].shape[0]}] : Training data DNS flows \n[{data["test_flat"].shape[0]}] : Test data DNS flows ')
    
    print(f'\nData description bucket based:\n{"=" * 40}')
    print(f'[{data["train_bucket"].shape[0]}] : Training bucket data DNS flows \n[{data["test_bucket"].shape[0]}] : Test bucket data DNS flows\n')
    data['train_bucket'].to_csv('/home/mpaul/projects/mpaul/mai/results/train_bucket.csv', index=False)
    data['test_bucket'].to_csv('/home/mpaul/projects/mpaul/mai/results/test_bucket.csv', index=False)
    return data

def get_model_handle(data=None):
    return DNSAttackModel(
            train_data=data['train_bucket'] if TYPE == 'bucket' else data['train_flat'],
            test_data=data['test_bucket'] if TYPE == 'bucket' else data['test_flat'],
            target_labels=target_labels,
            config_path='anomly_detection/config.yaml',
            target_features=bucket_target_features if TYPE == 'bucket' else flat_target_features,
            result_path=RESULT
        )

def main():
    # Prepare the data for training and testing
    data = prepare_data()

    # Get the model handle based on the prepared data
    model_handle = get_model_handle(data=data)
        
    # Train the model and get the trained pipeline
    pipe = model_handle.train_model()

    
    
    # Test the model and get the predicted DataFrame and classification report
    predicted_df, classification_report = model_handle.test_model(pipe)

    # Print the classification report
    print(f'\nClassification report:\n{"=" * 40}\n{classification_report}')
    classification_report.to_csv(f'/home/mpaul/projects/mpaul/mai/results/classification_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv', index=False)
    # Generate a detailed report based on the model's performance
    model_handle.generate_report(pipe)
    
    # Create a DNSAttackReport object to generate an attack report
    dns_attack_report = DNSAttackReport(predicted_df, data, TYPE, time_window=ATTACK_INTERVAL)
    
    # Generate the attack report
    dns_attack_report.generate_attack_report()
    
    
if __name__ == '__main__':
    main()