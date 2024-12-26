from elasticsearch import Elasticsearch
from config import ES_HOST

class ElasticsearchClient:
    def __init__(self):
        self.es_client = Elasticsearch([ES_HOST])


    def store_to_elasticsearch(self, data):
        """Store processed data to Elasticsearch"""
            # returnd
            
        try:
            index_name = f"dns_flows"
            data_dict = data.to_dict(orient='records')
            
            for record in data_dict:
                res = self.es_client.index(index=index_name, document=record)
                print(res)
            print(f"Successfully indexed {len(data_dict)} documents")

        except Exception as e:
            print(f"Error storing to Elasticsearch: {e}") 