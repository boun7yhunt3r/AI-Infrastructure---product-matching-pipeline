import faiss
import numpy as np
from typing import List, Optional

class VectorDB:
    def __init__(self, dimension: int = 512):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.id_to_product = {}
        self.next_id = 0
    
    def add_embedding(self, embedding: np.ndarray, product_id: str) -> int:
        if len(embedding) != self.dimension:
            raise ValueError(f"Embedding must be of dimension {self.dimension}")
 
        if len(embedding.shape) == 1:
            embedding = embedding.reshape(1, -1)
        

        self.index.add(embedding)
        

        internal_id = self.next_id
        self.id_to_product[internal_id] = product_id
        self.next_id += 1
        
        return internal_id
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[tuple[str, float]]:
        if len(query_embedding) != self.dimension:
            raise ValueError(f"Query embedding must be of dimension {self.dimension}")
        
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        distances, indices = self.index.search(query_embedding, k)
        
        results = []
        for i in range(len(indices[0])):
            internal_id = indices[0][i]
            distance = distances[0][i]
            product_id = self.id_to_product.get(internal_id, None)
            if product_id:
                results.append((product_id, float(distance)))
        
        return results
    
    def reset(self):
        self.index = faiss.IndexFlatL2(self.dimension)
        self.id_to_product = {}
        self.next_id = 0