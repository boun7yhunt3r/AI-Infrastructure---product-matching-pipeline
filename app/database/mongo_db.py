from pymongo import MongoClient
from typing import Dict, Any, List
import os

class MongoDB:
    def __init__(self, uri: str):
        self.client = MongoClient(uri)
        self.db = self.client["product_database"]
        self.products = self.db["products"]
        self.logs = self.db["logs"]
    
    def insert_product(self, product: Dict[str, Any]) -> str:
        result = self.products.insert_one(product)
        return str(result.inserted_id)
    
    def get_product(self, product_id: str) -> Dict[str, Any]:
        return self.products.find_one({"_id": product_id})
    
    def get_all_products(self) -> List[Dict[str, Any]]:
        return list(self.products.find())
    
    def log_event(self, log_data: Dict[str, Any]) -> str:
        result = self.logs.insert_one(log_data)
        return str(result.inserted_id)
    
    def close(self):
        self.client.close()