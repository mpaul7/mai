from nfstream import NFStreamer
import pandas as pd
import time
from threading import Thread
import queue
import netifaces
import os
from data_processor import DataProcessor
from config import INTERFACE, CAPTURE_INTERVAL

class NetworkMonitor:
    def __init__(self, interface_name, capture_interval=60):
        self.interface_name = self._validate_interface(interface_name)
        self.capture_interval = capture_interval
        self.data_queue = queue.Queue()
        self.data_processor = DataProcessor()
        
    def _validate_interface(self, interface_name):
        """Validate if the network interface exists"""
        available_interfaces = netifaces.interfaces()
        if interface_name not in available_interfaces:
            print(f"Warning: Interface {interface_name} not found!")
            print(f"Available interfaces: {available_interfaces}")
            raise ValueError(f"Invalid interface: {interface_name}")
        return interface_name
        
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
                flows.append(self._extract_flow_data(flow))
                    
        except PermissionError:
            print("Permission denied. Please run with sudo privileges.")
            raise
        except Exception as e:
            print(f"NFStreamer error: {str(e)}")
            print(f"Error type: {type(e)}")
            print("Please ensure you have sufficient permissions and the interface is up")
            raise
            
        return flows
        
    def _extract_flow_data(self, flow):
        """Extract relevant data from network flow"""
        return {
            'sip': flow.src_ip,
            'sport': flow.src_port,
            'dip': flow.dst_ip,
            'dport': flow.dst_port,
            'proto': flow.protocol,
            'first_timestamp': flow.bidirectional_first_seen_ms,
            'pkt_fwd_count': flow.bidirectional_packets,
            'pkt_bwd_count': flow.bidirectional_packets
        }
        
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
        # Check if interface is up
        try:
            addr = netifaces.ifaddresses(self.interface_name)
            if netifaces.AF_INET not in addr:
                print(f"Warning: Interface {self.interface_name} might not be up or doesn't have an IPv4 address")
        except Exception as e:
            print(f"Error checking interface status: {e}")

        capture_thread = Thread(target=self.continuous_capture, daemon=True)
        process_thread = Thread(target=self.process_data, daemon=True)
        
        print(f"Starting continuous capture on interface [{self.interface_name}]...")
        capture_thread.start()
        process_thread.start()
        
        return capture_thread, process_thread 