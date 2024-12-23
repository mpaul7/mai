# Elasticsearch and Kibana Setup on Ubuntu

This document provides instructions for starting and checking the status of Elasticsearch and Kibana on Ubuntu Linux.

## Resources
- https://pastebin.pl/view/b3102120
## Prerequisites

- Ensure you have Java installed (for Elasticsearch).
- Download and install Elasticsearch and Kibana from the [Elastic website](https://www.elastic.co/downloads/).

## Starting Elasticsearch

1. **Start Elasticsearch**:
   ```bash
   sudo systemctl start elasticsearch
   ```

2. **Enable Elasticsearch on Boot**:
   ```bash
   sudo systemctl enable elasticsearch
   ```

3. **Check Status**:
   ```bash
   sudo systemctl status elasticsearch
   ```

4. **Verify Elasticsearch is Running**:
   Open a terminal and run:
   ```bash
   curl -X GET "http://localhost:9200/"
   ```

## Starting Kibana

1. **Start Kibana**:
   ```bash
   sudo systemctl start kibana
   ```

2. **Enable Kibana on Boot**:
   ```bash
   sudo systemctl enable kibana
   ```

3. **Check Status**:
   ```bash
   sudo systemctl status kibana
   ```

## Accessing Kibana

Once Kibana is running, you can access it by opening a web browser and navigating to: