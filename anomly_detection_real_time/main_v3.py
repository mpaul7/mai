import threading
import queue
import time
from nfstream import NFStreamer
import pandas as pd
import numpy as np
from data_preparation import DataPreparation
from dns_attack_report import DNSAttackReport
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import joblib

# Optimized continuous capture function
def continuous_capture(interface, data_queue, interval, stop_event):
    """
    Continuously capture network traffic on specified interface.
    """
    try:
        while not stop_event.is_set():
            start_time = time.time()
            print(f"\nStarting new capture window at {time.strftime('%H:%M:%S')}")

            try:
                # Create a streamer instance
                streamer = NFStreamer(
                    source=interface,
                    active_timeout=0,
                    idle_timeout=0,
                    accounting_mode=1
                )

                flows = []

                # Collect flows for the specified interval
                for flow in streamer:
                    if time.time() - start_time >= interval:
                        break

                    pkt_fwd_count = np.random.choice([1, 2, 3, 10, 0, 4, 8, 5, 6, 7, 11, 9, 12, 64, 62, 13])
                    pkt_bwd_count = np.random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 10])

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

                if not flows_df.empty:
                    data_queue.put(flows_df)
                    print(f"Captured {len(flows_df)} flows in {interval} seconds")
                else:
                    print("No flows captured in this interval")

            except Exception as e:
                print(f"Error during capture: {e}")
                time.sleep(1)

    finally:
        print("Capture thread cleaning up resources.")

# Optimized processing function
def process_data(data_queue, stop_event):
    """
    Process the captured data from the queue.
    """
    try:
        while not stop_event.is_set():
            try:
                flows_df = data_queue.get(timeout=1)

                # Data preparation and prediction
                data_preparation = DataPreparation(df=flows_df)
                test_bucket, test_flat_bucket = data_preparation.bucketize_data(bucket_size=3, df=flows_df)
                test_flat_bucket['bucket'] = test_flat_bucket['bucket'].fillna(0)

                pipe = joblib.load('/home/solana/projects/mai/results/dns_attack_model.joblib')
                X_test = test_bucket[['pkt_flow_count_ratio']]

                y_pred = pipe.predict(X_test)
                y_pred_labels = np.where(y_pred == -1, 'dns_attack', 'dns')
                test_bucket['predicted_label'] = y_pred_labels

                attack_buckets = test_bucket[test_bucket['predicted_label'] == 'dns_attack']['bucket'].unique()
                non_attack_buckets = test_bucket[test_bucket['predicted_label'] == 'dns']['bucket'].unique()

                # Label flows
                for bucket in attack_buckets:
                    test_flat_bucket.loc[test_flat_bucket['bucket'] == bucket, 'label'] = 'dns_attack'

                for bucket in non_attack_buckets:
                    test_flat_bucket.loc[test_flat_bucket['bucket'] == bucket, 'label'] = 'dns'

                print(test_flat_bucket.groupby('label').size())
                test_flat_bucket = test_flat_bucket.head(25)
                print(test_flat_bucket.groupby('label').size())

                # Elasticsearch indexing
                es_data = [
                    {
                        '@timestamp': pd.to_datetime(row['first_timestamp'], unit='s'),
                        'source_ip': row['sip'],
                        'source_port': row['sport'],
                        'destination_ip': row['dip'],
                        'destination_port': row['dport'],
                        'protocol': row['proto'],
                        'packets_forward': row['pkt_fwd_count'],
                        'packets_backward': row['pkt_bwd_count'],
                        'bucket': int(row['bucket']),
                        'label': row['label']
                    }
                    for _, row in test_flat_bucket.iterrows()
                ]

                try:
                    es = Elasticsearch(['http://localhost:9200'])
                    actions = [
                        {
                            '_index': 'dns_flows',
                            '_source': doc
                        } for doc in es_data
                    ]
                    success, _ = bulk(es, actions)
                    print(f"Successfully indexed {success} documents")

                except Exception as e:
                    print(f"Error connecting to Elasticsearch: {e}")

            except queue.Empty:
                pass

            except Exception as e:
                print(f"Error during processing: {e}")

    finally:
        print("Processing thread cleaning up resources.")

if __name__ == "__main__":
    interface = "eno1"
    capture_interval = 15
    stop_event = threading.Event()

    # Create a queue with limited size
    data_queue = queue.Queue(maxsize=100)

    # Create threads
    capture_thread = threading.Thread(
        target=continuous_capture,
        args=(interface, data_queue, capture_interval, stop_event),
        daemon=True
    )

    process_thread = threading.Thread(
        target=process_data,
        args=(data_queue, stop_event),
        daemon=True
    )

    print(f"Starting continuous capture on interface [{interface}]...")
    capture_thread.start()
    process_thread.start()

    try:
        while not stop_event.is_set():
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping capture...")
        stop_event.set()

    capture_thread.join()
    process_thread.join()
    print("All threads stopped.")

