
import warnings
warnings.filterwarnings("ignore")   


import time
from network_monitor import NetworkMonitor
from config import INTERFACE, CAPTURE_INTERVAL
import queue
from threading import Thread
from elasticsearch_client import ElasticsearchClient
from process_data import ProcessData

def main():
    data_queue = queue.Queue()
    es_client = ElasticsearchClient()
    monitor = NetworkMonitor(INTERFACE, data_queue, CAPTURE_INTERVAL)
    processor = ProcessData(data_queue, es_client)
    capture_thread = monitor.start_monitoring() 
    process_thread = Thread(target=processor.process_data, daemon=True)
    process_thread.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping capture...")

if __name__ == "__main__":
    main()