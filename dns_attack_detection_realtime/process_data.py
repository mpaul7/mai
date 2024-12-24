import time
from inference import DataProcessor


class ProcessData:
    def __init__(self, data_queue, es_client):
        self.data_queue = data_queue
        self.data_processor = DataProcessor()
        self.es_client = es_client

    def process_data(self):
        """Process the captured data from the queue"""
        while True:
            try:
                flows_df = self.data_queue.get()
                test_bucket, test_flat_bucket = self.data_processor.prepare_data(flows_df)
                
                if test_bucket is not None and test_flat_bucket is not None:
                    predicted_labels = self.data_processor.predict_traffic(test_bucket, test_flat_bucket)

                
                if self.es_client:
                    self.es_client.store_to_elasticsearch(predicted_labels)

                    
            except Exception as e:
                print(f"Error in processing: {e}")
                time.sleep(1)