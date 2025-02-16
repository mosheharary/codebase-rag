from pinecone.grpc import PineconeGRPC, GRPCClientConfig
from pinecone import ServerlessSpec
import time
import os

class PineconeClient:
    def __init__(self, api_key="pclocal", host="http://localhost:5080", logger=None):
        """
        Initialize a Pinecone client.
        
        Args:
            api_key (str): API key for Pinecone (default: "pclocal")
            host (str): Host address of the Pinecone instance (default: "http://localhost:5080")
        """
        self.logger = logger
        self.client = PineconeGRPC(api_key=api_key, host=host)
        self.create_index(os.getenv("PINECONE_INDEX"), dimension=1536)        
        
    def create_index(self, name, dimension, metric="cosine", cloud="aws", region="us-east-1"):
        """
        Create a new index if it doesn't exist.
        
        Args:
            name (str): Name of the index
            dimension (int): Dimension of the vectors
            metric (str): Distance metric to use (default: "cosine")
            cloud (str): Cloud provider (default: "aws")
            region (str): Cloud region (default: "us-east-1")
            
        Returns:
            bool: True if index was created, False if it already existed
        """
        if not self.client.has_index(name):
            self.logger.info(f"Creating pinecone index {name}")
            self.client.create_index(
                name=name,
                dimension=dimension,
                metric=metric,
                spec=ServerlessSpec(
                    cloud=cloud,
                    region=region,
                )
            )
            return True
        else:
            self.logger.info(f"Pinecone Index {name} already exists")
        return False
    
    def wait_for_index_ready(self, index_name, check_interval=1):
        """
        Wait until the index is ready.
        
        Args:
            index_name (str): Name of the index
            check_interval (int): Seconds to wait between checks (default: 1)
        """
        while not self.client.describe_index(index_name).status['ready']:
            time.sleep(check_interval)
    
    def get_index(self, index_name, secure=False):
        """
        Get a reference to an index.
        
        Args:
            index_name (str): Name of the index
            secure (bool): Whether to use TLS (default: False)
            
        Returns:
            Index: Pinecone index object
        """
        index_host = self.client.describe_index(index_name).host
        return self.client.Index(
            host=index_host, 
            grpc_config=GRPCClientConfig(secure=secure)
        )
    
    def delete_index(self, index_name):
        """
        Delete an index.
        
        Args:
            index_name (str): Name of the index to delete
        """
        self.client.delete_index(index_name)
    
    def has_index(self, index_name):
        """
        Check if an index exists.
        
        Args:
            index_name (str): Name of the index
            
        Returns:
            bool: True if the index exists, False otherwise
        """
        return self.client.has_index(index_name)
    
    def describe_index(self, index_name):
        """
        Get information about an index.
        
        Args:
            index_name (str): Name of the index
            
        Returns:
            object: Index description
        """
        return self.client.describe_index(index_name)
    
    def upsert(self, index_name, vectors, namespace=None, batch_size=100, secure=False, 
               default_path=None, default_summary=None):
        """
        Upsert vectors into an index with optional default metadata fields.
        
        Args:
            index_name (str): Name of the index
            vectors (list): List of vector dictionaries with 'id', 'values', and optional 'metadata'
            namespace (str, optional): Namespace for the vectors
            batch_size (int): Number of vectors to upsert in each batch (default: 100)
            secure (bool): Whether to use TLS for the index connection (default: False)
            default_name (str, optional): Default name to add to metadata if not present
            default_path (str, optional): Default path to add to metadata if not present
            default_summary (str, optional): Default summary to add to metadata if not present
            
        Returns:
            dict: Upsert response from the last batch
        """
        try:
            index = self.get_index(index_name, secure=secure)
            id=default_path
            response = index.upsert(
                vectors=[
                    {
                        "id": id,
                        "values": vectors,
                        "metadata": 
                        {
                            'summary': default_summary
                        }
                    }
                ]
            )
            
            return response
        except Exception as e:
            self.logger.error(f"Error upserting vectors: {str(e)}")
            return None
    
    def search(self, index_name, vector, filter=None, top_k=10, namespace=None, 
               include_values=False, include_metadata=False, secure=False):
        """
        Search for similar vectors in an index.
        
        Args:
            index_name (str): Name of the index
            vector (list): Query vector
            filter (dict, optional): Metadata filter
            top_k (int): Number of results to return (default: 10)
            namespace (str, optional): Namespace to search in
            include_values (bool): Whether to include vector values in results (default: False)
            include_metadata (bool): Whether to include metadata in results (default: False)
            secure (bool): Whether to use TLS for the index connection (default: False)
            
        Returns:
            dict: Query response containing matches
        """
        index = self.get_index(index_name, secure=secure)
        
        response = index.query(
            vector=vector,
            filter=filter,
            top_k=top_k,
            include_values=include_values,
            include_metadata=include_metadata,
            namespace=namespace
        )
        
        return response