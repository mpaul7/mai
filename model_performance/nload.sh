#!/bin/bash

# Define parameters
INTERFACE="enp3s0f0"
OUTPUT_FILE="nload_log.txt"
DURATION=60          # Total duration in seconds (300 seconds = 5 minutes)
INTERVAL_MS=5000      # Refresh interval in milliseconds (5000 ms = 5 seconds)

echo "Monitoring network traffic on $INTERFACE in Gbits/s for $DURATION seconds..."
echo "Logging output to $OUTPUT_FILE"

# Run nload for 300 seconds; nload will refresh every 5 seconds.
# nload -t 100 -u g -U G devices enp1s0f0np0
timeout $DURATION nload -u G -t $INTERVAL_MS devices "$INTERFACE" 

echo "Monitoring complete. Data saved to $OUTPUT_FILE."
