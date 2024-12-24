# Network Configuration
INTERFACE = "enp0s31f6"
CAPTURE_INTERVAL = 15

# Model Configuration
MODEL_PATH = '/home/mpaul/projects/mpaul/mai/results/dns_attack_model.joblib'

# Elasticsearch Configuration
ES_HOST = 'http://localhost:9200'

# Data Preparation Configuration
BUCKET_SIZE = 3  # Set the bucket size for data preparation

# Target Features Configuration
TARGET_FEATURES = [
    'pkt_flow_count_ratio'
]  # Add your target features here