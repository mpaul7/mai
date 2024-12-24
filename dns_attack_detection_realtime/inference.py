
import joblib
import numpy as np

from data_preparation import DataPreparation
from config import MODEL_PATH, BUCKET_SIZE


class DataProcessor:
    def __init__(self):
        self.model_path = MODEL_PATH
            
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
            
    def predict_traffic(self, test_bucket, test_flat_bucket):
        """Analyze traffic using the loaded model"""
        try:
            X_test = test_bucket[['pkt_flow_count_ratio']]

            pipe = joblib.load(self.model_path)
            predictions = pipe.predict(X_test)
            labeled_data = self._label_flows(test_bucket, test_flat_bucket, predictions)
                
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
            
        TYPE = 'bucket'
        attack_buckets = predicted_df[predicted_df['predicted_label'] == 'dns_attack']['bucket'].unique()
        non_attack_buckets = predicted_df[predicted_df['predicted_label'] == 'dns']['bucket'].unique()
            
            
        non_attack_flows = []
        for bucket in non_attack_buckets:
            test_flat_bucket.loc[test_flat_bucket['bucket'] == bucket, 'label'] = 'dns'
                
        attack_flows = []
        for bucket in attack_buckets:
            test_flat_bucket.loc[test_flat_bucket['bucket'] == bucket, 'label'] = 'dns_attack'
        return test_flat_bucket
        