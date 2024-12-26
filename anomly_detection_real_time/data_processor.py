from elasticsearch import Elasticsearch

def _init_elasticsearch(self):
    return Elasticsearch(
        hosts=[{'host': 'YOUR-DOMAIN-ENDPOINT', 'port': 443}],
        http_auth=('master-user', 'master-password'),  # If using fine-grained access
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection
    ) 