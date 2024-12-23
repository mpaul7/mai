from nfstream import NFStreamer
import pandas as pd
import time
from threading import Thread
import queue
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.svm import OneClassSVM
from datetime import datetime
from io import BytesIO
import numpy as np
from data_preparation import DataPreparation
from dns_attack_report import DNSAttackReport
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

def continuous_capture(interface_name, data_queue, interval=60):
    """
    Continuously capture network traffic on specified interface
    """
    while True:
        try:
            start_time = time.time()
            print(f"\nStarting new capture window at {time.strftime('%H:%M:%S')}")
            
            # Create a new streamer instance for each capture window
            streamer = NFStreamer(
                source=interface_name,
                active_timeout=0,
                idle_timeout=0,
                accounting_mode=1)
            
            flows = []
            
            # Collect flows for the specified interval
            for flow in streamer:
                if time.time() - start_time >= interval:
                    del streamer  # Properly cleanup the streamer
                    break
                simulated_fwd_bwd_count = {
                1: [0, 1, 2],  2: [0, 1, 2, 3],  3: [0, 1, 2, 3], 
                10: [10],  0: [0, 2, 3, 4, 5, 6, 7, 8, 10],  4: [2, 3, 4],  8: [8], 
                5: [4, 5],  6: [6],  7: [0, 7], 11: [11],  
                9: [9], 12: [12], 64: [64], 62: [62], 13: [13]
                }
                pkt_fwd_count_list = [ 1,  2,  3, 10,  0,  4,  8,  5,  6,  7, 11,  9, 12, 64, 62, 13]
                pkt_fwd_count = pkt_fwd_count_list[int(len(pkt_fwd_count_list) * np.random.random())]
                pkt_bwd_key_values = simulated_fwd_bwd_count[pkt_fwd_count]
                pkt_bwd_count = pkt_bwd_key_values[int(len(pkt_bwd_key_values) * np.random.random())]
                flows.append({
                    'sip': flow.src_ip,
                    'sport': flow.src_port,
                    'dip': flow.dst_ip,
                    'dport': flow.dst_port,
                    'proto': flow.protocol,
                    'first_timestamp': flow.bidirectional_first_seen_ms,
                    'pkt_fwd_count': pkt_fwd_count,
                    'pkt_bwd_count': pkt_bwd_count
                })
            
            # Convert to DataFrame
            flows_df = pd.DataFrame(flows)
            # flows_df = streamer
            flows_df.to_csv('flows.csv', index=False)
            if not flows_df.empty:
                data_queue.put(flows_df)
                print(f"Captured {len(flows_df)} flows in {interval} seconds")
            else:
                print("No flows captured in this interval")
                
        except Exception as e:
            print(f"Error in capture: {e}")
            time.sleep(1)

def process_data(data_queue):
    """
    Process the captured data from the queue
    
    Args:
        data_queue (Queue): Queue containing captured network flow data
    """
    while True:
        try:
            # Get the next batch of data
            _test_df = data_queue.get()
            
            # col_rename = {'src_ip': 'sip', 'dst_ip': 'dip', 
            #               'src_port': 'sport', 'dst_port': 'dport',
            #               'src2dst_packets': 'pkt_fwd_count', 'dst2src_packets': 'pkt_bwd_count'}
            # output_col = ['sip', 'sport', 'dip', 'dport', 'protocol', 'pkt_fwd_count', 'pkt_bwd_count']
            # flows_df.rename(columns=col_rename, inplace=True)
            # flows_df = flows_df[output_col]
            # print(flows_df.head())
            data_preparation = DataPreparation(df=_test_df)
            test_bucket, test_flat_bucket = data_preparation.bucketize_data(bucket_size=3, df=_test_df)
            test_flat_bucket['bucket'] = test_flat_bucket['bucket'].fillna(0)
            # print(test_bucket, 222)
            pipe_path = '/home/mpaul/projects/mpaul/mai/results/dns_attack_model.joblib'
            pipe = joblib.load(pipe_path)
            
            X_test = test_bucket[['pkt_flow_count_ratio']]
            
            # print(test_bucket, 333)
            y_pred = pipe.predict(X_test)
            # print(y_pred)
            target_labels = ['dns', 'dns_attack']
            y_pred_labels = np.where(y_pred == -1, target_labels[1], target_labels[0])
            test_bucket['predicted_label'] = y_pred_labels
            predicted_df = test_bucket
            
            # print(test_bucket.predicted_label)
            TYPE = 'bucket'
            # dns_attack_report = DNSAttackReport(test_bucket, _test_df, TYPE, time_window=30)
            # dns_attack_report.generate_attack_report()
            
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
            test_flat_bucket = test_flat_bucket.head(25)
            print(test_flat_bucket.groupby('label').size())
            # Prepare data for Elasticsearch
            es_data = []
            for _, row in test_flat_bucket.iterrows():
                doc = {
                    '@timestamp': pd.to_datetime(row['first_timestamp'], unit='s'),
                    'source_ip': row['sip'],
                    'source_port': row['sport'], 
                    'destination_ip': row['dip'],
                    'destination_port': row['dport'],
                    'protocol': row['proto'],
                    'packets_forward': row['pkt_fwd_count'],
                    'packets_backward': row['pkt_bwd_count'],
                    # 'payload_forward': row['pl_fwd_count'],
                    # 'payload_backward': row['pl_bwd_count'],
                    # 'payload_bytes_forward': row['pl_len_fwd_total'],
                    # 'payload_bytes_backward': row['pl_len_bwd_total'],
                    # 'packet_bytes_forward': row['pkt_len_fwd_total'], 
                    # 'packet_bytes_backward': row['pkt_len_bwd_total'],
                    'bucket': int(row['bucket']),
                    'label': row['label']
                }
                es_data.append(doc)

            # Connect to Elasticsearch
            try:
                es = Elasticsearch(['http://localhost:9200'])
                
                # Bulk index the documents
                actions = [
                    {
                        '_index': 'dns_flows',
                        '_source': doc
                    }
                    for doc in es_data
                ]
                
                success, failed = bulk(es, actions)
                print(f"Successfully indexed {success} documents")
                if failed:
                    print(f"Failed to index {len(failed)} documents")
                    
            except Exception as e:
                print(f"Error connecting to Elasticsearch: {e}")
            
            
            # if len(non_attack_flows) > 0:
            #     non_attack_flows = pd.concat(non_attack_flows)
            # if len(attack_flows) > 0    :
            #     attack_flows = pd.concat(attack_flows)
            # total_flows = pd.concat([non_attack_flows, attack_flows])
            # print(total_flows.head())
            # attack_flows = pd.concat(attack_flows)
            # print(non_attack_flows.head())
            
            # attack_flows = []
            # for bucket in attack_buckets:
            #     bucket_flows = self.data['test_flat_bucket'][self.data['test_flat_bucket']['bucket'] == bucket]
            #     bucket_suspicious_flows = bucket_flows[
            #         # Check for high number of responses compared to requests 
            #         ((bucket_flows['pl_bwd_count'] / bucket_flows['pl_fwd_count']) > 1) 
            #     ]
            #     attack_flows.append(bucket_suspicious_flows)
            
            # print(test_bucket.head())
            # print(test_flat_bucket.head())
            
            # if not flows_df.empty:
                # # Here you can add your call to data_preparation module
                # # Example: data_preparation.process_flows(flows_df)
                
                # print("\nFlow Statistics:")
                # print(f"Unique source IPs: {flows_df['src_ip'].nunique()}")
                # print(f"Unique destination IPs: {flows_df['dst_ip'].nunique()}")
                # print(f"Protocols used: {flows_df['protocol'].unique()}")
                
                # # Save to CSV with timestamp
                # timestamp = time.strftime("%Y%m%d-%H%M%S")
                # flows_df.to_csv(f'captured_traffic_{timestamp}.csv', index=False)
                
        except Exception as e:
            print(f"Error in processing: {e}")
            time.sleep(1)

if __name__ == "__main__":
    interface = "enp0s31f6"  
    capture_interval = 15
    
    # Create a queue for communication between threads
    data_queue = queue.Queue()
    
    # Create and start capture thread
    capture_thread = Thread(target=continuous_capture, 
                        args=(interface, data_queue, capture_interval),
                        daemon=True)
    
    # Create and start processing thread
    process_thread = Thread(target=process_data, 
                        args=(data_queue,),
                        daemon=True)
    
    print(f"Starting continuous capture on interface [{interface}]...")
    capture_thread.start()
    process_thread.start()
    
    try:
        # Keep main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping capture...")