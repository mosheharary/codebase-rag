from opensearchpy import OpenSearch, RequestError, NotFoundError
import ssl
import numpy as np

class OpensearchClient:
    def __init__(self, hosts=['localhost:9200'], logger=None):
        # OpenSearch client
        self.client = OpenSearch(
            hosts=hosts,
            http_auth=(os.getenv("OPENSEARCH_USER"), os.getenv("OPENSEARCH_PASSWORD")),
            use_ssl=True,
            verify_certs=False,
            ssl_show_warn=False
        )
        
        # Index names for different search methods
        self.vector_index = os.getenv("OPENSEARCH_VECTOR_INDEX")
        self.bm25_index = os.getenv("OPENSEARCH_BM25_INDEX")
        self.logger = logger
        self.create_indices(recreate=False)

    def create_indices(self, recreate=False):
        # Common settings for both indices
        common_settings = {
            "analysis": {
                "normalizer": {
                    "path_normalizer": {
                        "type": "custom",
                        "char_filter": [],
                        "filter": ["lowercase", "asciifolding"]
                    }
                }
            }
        }

        # Vector Index Configuration
        self.vector_index_body = {
            "settings": {
                "index.knn": True,
                **common_settings
            },
            "mappings": {
                "properties": {
                    "vector_field": {
                        "type": "knn_vector",
                        "dimension": 1536,
                        "method": {
                            "name": "hnsw",
                            "space_type": "l2",
                            "engine": "nmslib"
                        }
                    },
                    "path": {
                        "type": "keyword",
                        "normalizer": "path_normalizer",
                        "fields": {
                            "raw": {
                                "type": "keyword"
                            }
                        }
                    },
                    "metadata": {
                        "type": "object",
                        "properties": {
                            "summary": {"type": "text"},
                            "md5": {"type": "keyword"}
                        }
                    }
                }
            }
        }

        # BM25 Index Configuration
        self.bm25_index_body = {
            "settings": {
                **common_settings,  # Including common settings with normalizer
                "analysis": {  # Merging analysis settings here
                    "normalizer": {
                        "path_normalizer": {
                            "type": "custom",
                            "char_filter": [],
                            "filter": ["lowercase", "asciifolding"]
                        }
                    },
                    "analyzer": {
                        "default": {
                            "type": "standard"
                        }
                    }
                }
            },
            "mappings": {
                "properties": {
                    "path": {
                        "type": "keyword",
                        "normalizer": "path_normalizer",
                        "fields": {
                            "raw": {
                                "type": "keyword"
                            }
                        }
                    },
                    "metadata": {
                        "type": "object",
                        "properties": {
                            "summary": {
                                "type": "text",
                                "analyzer": "standard",
                                "search_analyzer": "standard"
                            },
                            "md5": {"type": "keyword"}
                        }
                    }
                }
            }
        }

        for index_name in [self.vector_index, self.bm25_index]:
            if recreate and self.client.indices.exists(index=index_name):
                # Delete existing index if recreation is requested
                self.client.indices.delete(index=index_name)

            # Only create index if it doesn't already exist
            if not self.client.indices.exists(index=index_name):
                # Choose the appropriate index configuration
                index_body = (
                    self.vector_index_body if index_name == self.vector_index 
                    else self.bm25_index_body
                )

                try:
                    self.client.indices.create(index=index_name, body=index_body)
                except Exception as e:
                    self.logger.error(f"Error creating index {index_name}: {e}")

                    

    def get_document_by_path(self, path, index_name):
        """
        Get document by exact path match.
        Returns (document, document_id) tuple or (None, None) if not found.
        """
        search_path = "path.raw"
        try:
            existing_doc_query = {
                "query": {
                    "term": {
                        search_path: path
                    }
                }
            }
            
            result = self.client.search(
                index=index_name,
                body=existing_doc_query
            )
            
            hits = result['hits']['hits']
            if len(hits) > 1:
                self.logger.error(f"Multiple documents found with path: {path}. This should not happen!")
                return hits[0]['_source'], hits[0]['_id']
            elif len(hits) == 1:
                #print (f"[{hits[0]['_id']}] {hits[0]['_source']['path']}")
                return hits[0]['_source'], hits[0]['_id']
            #print (f"[None] {path}")
            return None, None
            
        except NotFoundError:
            return None, None

    def index_document(self, path, metadata, vector):
        """
        Index document with guaranteed path uniqueness.
        Updates existing document if path exists, creates new one if it doesn't.
        """
        if not path:
            raise ValueError("Path cannot be empty")
            
        try:
            # Check both indices for existing documents
            vector_doc, vector_id = self.get_document_by_path(path, self.vector_index)
            bm25_doc, bm25_id = self.get_document_by_path(path, self.bm25_index)
            
            try:
                if vector_doc:
                    self.client.update(
                        index=self.vector_index,
                        id=vector_id,
                        body={
                            'doc': {
                                'path': path,
                                'vector_field': vector,
                                'metadata': metadata
                            }
                        }
                    )
                else:
                    # Document doesn't exist, create new
                    self.client.index(
                        index=self.vector_index,
                        body={
                            'path': path,
                            'vector_field': vector,
                            'metadata': metadata
                        }
                    )
            except Exception as e:
                self.logger.error(f"Error indexing document with path {path} to {self.vector_index}: {str(e)}")                
            
            try:
                if bm25_doc:
                    self.client.update(
                        index=self.bm25_index,
                        id=bm25_id,
                        body={
                            'doc': {
                                'path': path,
                                'metadata': metadata
                            }
                        }
                    )
                else:
                    # Document doesn't exist, create new
                    self.client.index(
                        index=self.bm25_index,
                        body={
                            'path': path,
                            'metadata': metadata
                        }
                    )
            except Exception as e:
                self.logger.error(f"Error indexing document with path {path} to {self.bm25_index}: {str(e)}")

        except Exception as e:
            self.logger.error(f"Error indexing document with path {path}: {str(e)}")
        return
        

    def hybrid_search(self, query, query_vector, k=5, vector_weight=0.5):
        """
        Perform a hybrid search combining vector similarity and BM25 search.
        Adapted for the new index structure with path-based uniqueness.
        """
        # Perform vector similarity search
        vector_search = self.client.search(
            index=self.vector_index,
            body={
                "size": k,
                # Remove the _source parameter to include all fields
                "query": {
                    "knn": {
                        "vector_field": {
                            "vector": query_vector,
                            "k": k
                        }
                    }
                }
            }
        )

        # Perform BM25 text search
        bm25_search = self.client.search(
            index=self.bm25_index,
            body={
                "size": k,
                "_source": True,  # Set to True to include all fields
                "query": {
                    "multi_match": {
                        "query": query,
                        "fields": ["metadata.summary"],  # Searchable fields
                        "type": "best_fields"
                    }
                }
            }
        )
        # Merge and deduplicate results using path as unique identifier
        merged_results = {}

        # Process vector search results
        for hit in vector_search['hits']['hits']:
            path = hit['_source']['path']
            metadata = hit['_source']['metadata']
            merged_results[path] = {
                'path': path,
                'metadata': metadata,
                'vector_score': hit['_score'],
                'bm25_score': 0
            }

        # Process BM25 search results
        for hit in bm25_search['hits']['hits']:
            path = hit['_source']['path']
            metadata = hit['_source']['metadata']

            if path in merged_results:
                # If document exists from vector search, combine scores
                merged_results[path]['bm25_score'] = hit['_score']
            else:
                # New document from BM25 search
                merged_results[path] = {
                    'path': path,
                    'metadata': metadata,
                    'vector_score': 0,
                    'bm25_score': hit['_score']
                }

        # Normalize scores
        max_vector_score = max((r['vector_score'] for r in merged_results.values()), default=1)
        max_bm25_score = max((r['bm25_score'] for r in merged_results.values()), default=1)

        # Calculate hybrid score and sort
        hybrid_results = []
        for result in merged_results.values():
            # Normalize scores before combining
            normalized_vector_score = result['vector_score'] / max_vector_score if max_vector_score > 0 else 0
            normalized_bm25_score = result['bm25_score'] / max_bm25_score if max_bm25_score > 0 else 0
            
            # Weighted combination of normalized scores
            hybrid_score = (
                vector_weight * normalized_vector_score + 
                (1 - vector_weight) * normalized_bm25_score
            )
            
            hybrid_results.append({
                'path': result['path'],
                'metadata': result['metadata'],
                'hybrid_score': hybrid_score,
                'vector_score': normalized_vector_score,
                'bm25_score': normalized_bm25_score
            })

        # Sort by hybrid score and return top k
        return sorted(
            hybrid_results, 
            key=lambda x: x['hybrid_score'], 
            reverse=True
        )[:k]
