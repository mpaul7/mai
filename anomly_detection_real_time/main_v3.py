import threading
import queue
import time
from nfstream import NFStreamer
import pandas as pd
import numpy as np
from data_preparation import DataPreparation
from dns_attack_report import DNSAttackReport
from elasticsearch import Elasticsearch, helpers
import joblib
import psutil
import gc
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename=f'dns_detection_{datetime.now().strftime("%Y%m%d")}.log'
)

def check_memory_usage():
    """Monitor memory usage of the current process"""
    process = psutil.Process()
    memory_info = process.memory_info()
    return memory_info.rss / 1024 / 1024  # Memory usage in MB

def cleanup_memory():
    """Force garbage collection"""
    gc.collect()

def document_generator(test_flat_bucket):
    """Generate documents for ES bulk insertion"""
    for _, row in test_flat_bucket.iterrows():
        yield {
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

def continuous_capture(interface, data_queue, interval, stop_event):
    """
    Continuously capture network traffic on specified interface.
    """
    try:
        while not stop_event.is_set():
            start_time = time.time()
            logging.info(f"Starting new capture window at {time.strftime('%H:%M:%S')}")

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

                    # Simulated packet counts for testing
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
                    logging.info(f"Captured {len(flows_df)} flows in {interval} seconds")
                else:
                    logging.warning("No flows captured in this interval")

            except Exception as e:
                logging.error(f"Error during capture: {e}")
                time.sleep(1)

    finally:
        logging.info("Capture thread cleaning up resources.")

def process_data(data_queue, stop_event):
    """
    Process the captured data from the queue with optimized memory usage.
    """
    memory_threshold = 1000  # MB
    batch_size = 500
    
    try:
        # Initialize Elasticsearch with optimized settings
        es = Elasticsearch(
            ['http://localhost:9200'],
            maxsize=25,
            timeout=30,
            retry_on_timeout=True,
            sniff_on_start=True,
            sniff_on_connection_fail=True
        )
        
        # Optimize index settings for bulk loading
        index_settings = {
            "settings": {
                "refresh_interval": "30s",
                "number_of_replicas": 0
            }
        }
        
        # Create index if it doesn't exist
        if not es.indices.exists(index='dns_flows'):
            es.indices.create(index='dns_flows', body={
                "mappings": {
                    "properties": {
                        "@timestamp": {"type": "date"},
                        "source_ip": {"type": "ip"},
                        "source_port": {"type": "integer"},
                        "destination_ip": {"type": "ip"},
                        "destination_port": {"type": "integer"},
                        "protocol": {"type": "integer"},
                        "packets_forward": {"type": "integer"},
                        "packets_backward": {"type": "integer"},
                        "bucket": {"type": "integer"},
                        "label": {"type": "keyword"}
                    }
                }
            })
        
        es.indices.put_settings(index='dns_flows', body=index_settings)

        while not stop_event.is_set():
            try:
                # Check memory usage
                current_memory = check_memory_usage()
                if current_memory > memory_threshold:
                    logging.warning(f"High memory usage ({current_memory:.2f}MB). Cleaning up...")
                    cleanup_memory()
                    time.sleep(2)  # Give system time to recover
                    continue

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

                # Label flows
                attack_buckets = test_bucket[test_bucket['predicted_label'] == 'dns_attack']['bucket'].unique()
                non_attack_buckets = test_bucket[test_bucket['predicted_label'] == 'dns']['bucket'].unique()

                for bucket in attack_buckets:
                    test_flat_bucket.loc[test_flat_bucket['bucket'] == bucket, 'label'] = 'dns_attack'
                for bucket in non_attack_buckets:
                    test_flat_bucket.loc[test_flat_bucket['bucket'] == bucket, 'label'] = 'dns'

                logging.info(f"Label distribution: {test_flat_bucket.groupby('label').size().to_dict()}")

                # Optimized Elasticsearch bulk insertion using streaming
                try:
                    success_count = 0
                    for ok, response in helpers.streaming_bulk(
                        es,
                        (
                            {
                                "_index": "dns_flows",
                                "_source": doc
                            }
                            for doc in document_generator(test_flat_bucket)
                        ),
                        chunk_size=batch_size,
                        max_retries=3,
                        yield_ok=False,
                        raise_on_error=False,
                        max_chunk_bytes=10485760  # 10MB chunk size
                    ):
                        if ok:
                            success_count += 1
                        else:
                            logging.error(f"Error in document: {response}")
                    
                    logging.info(f"Successfully indexed {success_count} documents")

                except Exception as e:
                    logging.error(f"Error during bulk insertion: {e}")

                # Clean up variables
                del flows_df, test_bucket, test_flat_bucket, X_test, y_pred, y_pred_labels
                cleanup_memory()

            except queue.Empty:
                pass

            except Exception as e:
                logging.error(f"Error during processing: {e}")
                time.sleep(1)

    finally:
        # Restore index settings
        restore_settings = {
            "settings": {
                "refresh_interval": "1s",
                "number_of_replicas": 1
            }
        }
        try:
            es.indices.put_settings(index='dns_flows', body=restore_settings)
        except:
            pass
        logging.info("Processing thread cleaning up resources.")

def main():
    interface = "eno1"  # Change this to match your network interface
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

    logging.info(f"Starting continuous capture on interface [{interface}]...")
    capture_thread.start()
    process_thread.start()

    try:
        while not stop_event.is_set():
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("\nStopping capture...")
        stop_event.set()

    capture_thread.join()
    process_thread.join()
    logging.info("All threads stopped.")

if __name__ == "__main__":
    main()

