from nfstream import NFStreamer
import pandas as pd
import numpy as np
import time
from threading import Thread
import queue
import netifaces
import os
from data_processor import DataProcessor
from config import INTERFACE, CAPTURE_INTERVAL

class NetworkMonitor:
    def __init__(self, interface_name, capture_interval=60):
        self.interface_name = interface_name
        self.capture_interval = capture_interval
        self.data_queue = queue.Queue()
        self.data_processor = DataProcessor()
        
    def continuous_capture(self):
        """Continuously capture network traffic on specified interface"""
        while True:
            try:
                start_time = time.time()
                print(f"\nStarting new capture window at {time.strftime('%H:%M:%S')}")
                
                # Check if running as root
                if os.geteuid() != 0:
                    print("Warning: This script may need to be run with sudo privileges")
                
                flows = self._capture_flows(start_time)
                
                if flows:
                    flows_df = pd.DataFrame(flows)
                    self.data_queue.put(flows_df)
                    print(f"Captured {len(flows_df)} flows in {self.capture_interval} seconds")
                else:
                    print("No flows captured in this interval")
                    
            except Exception as e:
                print(f"Error in capture: {e}")
                print("Waiting before retry...")
                time.sleep(5)  # Increased wait time before retry
                
    def _capture_flows(self, start_time):
        """Capture network flows using NFStreamer"""
        flows = []
        
        try:
            # Create streamer instance
            print(f"Attempting to create NFStreamer for interface: {self.interface_name}")
            streamer = NFStreamer(
                source=self.interface_name,
                active_timeout=0,
                idle_timeout=0,
                accounting_mode=1,
                snapshot_length=65535,  # Maximum snapshot length
                promiscuous_mode=True,
                statistical_analysis=False,  # Disable statistical analysis
                splt_analysis=False,  # Disable sequence of packet lengths and times
                n_dissections=0,  # Disable packet dissection
                max_nflows=0  # No limit on number of flows
            )
            
            print("NFStreamer instance created successfully")
            
            for flow in streamer:
                if time.time() - start_time >= self.capture_interval:
                    break
                """Extract relevant data from network flow"""
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

                _flow_data = {
                    'sip': flow.src_ip,
                    'sport': flow.src_port,
                    'dip': flow.dst_ip,
                    'dport': flow.dst_port,
                    'proto': flow.protocol,
                    'first_timestamp': flow.bidirectional_first_seen_ms,
                    'pkt_fwd_count': pkt_fwd_count,
                    'pkt_bwd_count': pkt_bwd_count
                }
                flows.append(_flow_data)
                    
        except PermissionError:
            print("Permission denied. Please run with sudo privileges.")
            raise
        except Exception as e:
            print(f"NFStreamer error: {str(e)}")
            print(f"Error type: {type(e)}")
            print("Please ensure you have sufficient permissions and the interface is up")
            raise
            
        return flows
            
    def process_data(self):
        """Process the captured data from the queue"""
        while True:
            try:
                flows_df = self.data_queue.get()
                test_bucket, test_flat_bucket = self.data_processor.prepare_data(flows_df)
                
                if test_bucket is not None and test_flat_bucket is not None:
                    self.data_processor.analyze_traffic(test_bucket, test_flat_bucket)
                    
            except Exception as e:
                print(f"Error in processing: {e}")
                time.sleep(1)
                
    def start_monitoring(self):
        """Start the monitoring threads"""

        capture_thread = Thread(target=self.continuous_capture, daemon=True)
        process_thread = Thread(target=self.process_data, daemon=True)
        
        print(f"Starting continuous capture on interface [{self.interface_name}]...")
        capture_thread.start()
        process_thread.start()
        
        return capture_thread, process_thread 