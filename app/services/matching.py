from typing import Dict, Any
from models.clip_model import CLIPModel
import numpy as np
from database.mongo_db import MongoDB
from database.vector_db import VectorDB
from utils.logging import Logger
import io

class ProductMatcher:
    def __init__(self, mongo_db: MongoDB, vector_db: VectorDB, triton_url: str, logger: Logger):
        self.mongo_db = mongo_db
        self.vector_db = vector_db
        self.model = CLIPModel(triton_url)
        self.logger = logger
        self._initialize_db()
    
    def _initialize_db(self):
        """Load products from MongoDB into vector DB"""
        products = self.mongo_db.get_all_products()
        for product in products:
            if "embedding" in product:
                embedding = np.array(product["embedding"])
                self.vector_db.add_embedding(embedding, str(product["_id"]))
    
    async def match_product(self, image_file) -> Dict[str, Any]:
        """Match an input image against stored products"""
        try:
            image_bytes = await image_file.read()
            

            embedding = self.model.get_image_embedding(image_bytes)
            
            matches = self.vector_db.search(embedding, k=1)
            if not matches:
                return {"match": None, "message": "No matches found"}
            
          
            product_id, distance = matches[0]
            product = self.mongo_db.get_product(product_id)
            
           
            self.logger.log_event({
                "type": "search",
                "query_image": image_file.filename,
                "matched_product": product_id,
                "distance": distance,
                "status": "success"
            })
            
            return {
                "match": product,
                "distance": distance,
                "status": "success"
            }
        except Exception as e:
            self.logger.log_event({
                "type": "error",
                "error": str(e),
                "status": "failed"
            })
            raise