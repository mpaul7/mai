import pandas as pd
import time
import joblib
from elasticsearch import Elasticsearch
from data_preparation import DataPreparation
from config import MODEL_PATH, ES_HOST, BUCKET_SIZE, TARGET_FEATURES
import numpy as np

class DataProcessor:
    def __init__(self):
        self.model_path = MODEL_PATH
        self.es_client = self._init_elasticsearch()
        
    def _init_elasticsearch(self):
        """Initialize Elasticsearch connection"""
        try:
            return Elasticsearch([ES_HOST])
        except Exception as e:
            print(f"Failed to connect to Elasticsearch: {e}")
            return None
            
    def prepare_data(self, df):
        """Prepare and analyze network traffic data"""
        try:
            data_preparation = DataPreparation(df=df)
            test_bucket, test_flat_bucket = data_preparation.bucketize_data(bucket_size=BUCKET_SIZE, df=df)
            test_flat_bucket['bucket'] = test_flat_bucket['bucket'].fillna(0)
            
            return test_bucket, test_flat_bucket
        except Exception as e:
            print(f"Error in data preparation: {e}")
            return None, None
            
    def analyze_traffic(self, test_bucket, test_flat_bucket):
        """Analyze traffic using the loaded model"""
        try:
            # test_bucket = test_bucket[TARGET_FEATURES ]
            X_test = test_bucket[['pkt_flow_count_ratio']]
            # print(X_test)

            pipe = joblib.load(self.model_path)
            predictions = pipe.predict(X_test)
            labeled_data = self._label_flows(test_bucket, test_flat_bucket, predictions)
            
            if self.es_client:
                self._store_to_elasticsearch(labeled_data)
                
            return labeled_data
        except Exception as e:
            print(f"Error in traffic analysis: {e}")
            return None
            
    def _label_flows(self, test_bucket, test_flat_bucket, predictions):
        """Label flows based on predictions and debucketize data"""
        # Define target labels
        target_labels = ['dns', 'dns_attack']
        
        # Create predicted labels based on predictions
        y_pred_labels = np.where(predictions == -1, target_labels[1], target_labels[0])
        test_bucket['predicted_label'] = y_pred_labels
        predicted_df = test_bucket
            
            # print(test_bucket.predicted_label)
        TYPE = 'bucket'
        attack_buckets = predicted_df[predicted_df['predicted_label'] == 'dns_attack']['bucket'].unique()
        non_attack_buckets = predicted_df[predicted_df['predicted_label'] == 'dns']['bucket'].unique()
            # print(attack_buckets)
            # print(non_attack_buckets)
            
            
        non_attack_flows = []
        for bucket in non_attack_buckets:
            test_flat_bucket.loc[test_flat_bucket['bucket'] == bucket, 'label'] = 'dns'
                # bucket_flows = test_flat_bucket[test_flat_bucket['bucket'] == bucket].assign(label='dns')
                # print(test_flat_bucket.head())
                # non_attack_flows.append(bucket_flows)
                
        attack_flows = []
        for bucket in attack_buckets:
            test_flat_bucket.loc[test_flat_bucket['bucket'] == bucket, 'label'] = 'dns_attack'
                # bucket_flows = test_flat_bucket[test_flat_bucket['bucket'] == bucket].assign(label='dns_attack')
                # print(test_flat_bucket.head())
                # attack_flows.append(bucket_flows)
        print(test_flat_bucket.shape)
        return test_flat_bucket
        
    def _store_to_elasticsearch(self, data):
        """Store processed data to Elasticsearch"""
            # returnd
            
        try:
            index_name = f"network_flows"
            data_dict = data.to_dict(orient='records')
            
            for record in data_dict:
                self.es_client.index(index=index_name, document=record)
        except Exception as e:
            print(f"Error storing to Elasticsearch: {e}") 