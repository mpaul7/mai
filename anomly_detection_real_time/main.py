import time
from network_monitor import NetworkMonitor
from config import INTERFACE, CAPTURE_INTERVAL

def main():
    monitor = NetworkMonitor(INTERFACE, CAPTURE_INTERVAL)
    capture_thread, process_thread = monitor.start_monitoring()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping capture...")

if __name__ == "__main__":
    main()