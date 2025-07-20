import os
import pandas as pd
import numpy as np
from PIL import Image
from io import BytesIO
import requests
import grpc
from app.database.mongo_db import MongoDB
from app.database.vector_db import VectorDB
from app.models.clip_model import CLIPModel
from tqdm import tqdm
import logging
import time
from tenacity import retry, stop_after_attempt, wait_exponential

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FashionDataProcessor:
    def __init__(self, mongo_uri, triton_url, image_folder, max_retries=3):
        self.mongo_db = MongoDB(mongo_uri)
        self.vector_db = VectorDB()
        self.max_retries = max_retries
        
       
        self.triton_url = triton_url
        
        try:
        
            self._check_triton_health_http()
            self.clip_model = CLIPModel(self.triton_url)
        except Exception as e:
            logger.error(f"Failed to connect to Triton server: {e}")
            raise RuntimeError("Could not initialize CLIP model - is Triton server running?")
        self.image_folder = image_folder

    def _check_triton_health_http(self):
        """Check if Triton server is healthy using HTTP"""
        try:
            url = f"http://{self.triton_url}/v2/health/ready"
            response = requests.get(url, timeout=10)
            if response.status_code != 200:
                raise RuntimeError(f"Triton server health check failed with status {response.status_code}")
            logger.info("Triton server health check passed")
        except Exception as e:
            logger.error(f"Triton health check failed: {e}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )
    def get_embedding_with_retry(self, image_bytes):
        """Get embedding with retry logic"""
        try:
            embedding = self.clip_model.get_image_embedding(image_bytes)
            if embedding is None:
                raise ValueError("Received null embedding from model")
            return embedding
        except Exception as e:
            logger.warning(f"Embedding generation failed, retrying... Error: {str(e)}")
            raise

    def process_excel(self, excel_path):
        """Process the Excel file and prepare database entries"""
        df = pd.read_csv(excel_path)
        failed_products = []
        
        for _, row in tqdm(df.iterrows(), total=len(df)):
            try:
                product_id = str(row['ProductId'])
                image_path = os.path.join(self.image_folder, f"{product_id}.jpg")
                
                if not os.path.exists(image_path):
                    logger.warning(f"Image not found for {product_id}")
                    continue
                
                
                with open(image_path, 'rb') as f:
                    image_bytes = f.read()
                
                try:
                    embedding = self.get_embedding_with_retry(image_bytes)
                except Exception as e:
                    logger.error(f"Failed to get embedding for {product_id} after retries: {str(e)}")
                    failed_products.append(product_id)
                    continue

                if embedding.dtype != np.float32:
                    embedding = embedding.astype(np.float32)
                    logger.debug(f"Converted embedding to FP32 for {product_id}")
                    
                # Prepare product document
                product = {
                    "_id": product_id,
                    "gender": row['Gender'],
                    "category": row['Category'],
                    "subcategory": row['SubCategory'],
                    "product_type": row['ProductType'],
                    "color": row['Colour'],
                    "usage": row['Usage'],
                    "title": row['ProductTitle'],
                    "image_url": row['ImageURL'],
                    "embedding": embedding.tolist()
                }
                
                self.mongo_db.insert_product(product)
                
        
                self.vector_db.add_embedding(embedding, product_id)
                
            except Exception as e:
                logger.error(f"Error processing {row['ProductId']}: {str(e)}")
                failed_products.append(row['ProductId'])
                
        if failed_products:
            logger.warning(f"Failed to process {len(failed_products)} products: {failed_products}")
        logger.info("Data processing completed")

if __name__ == "__main__":
    processor = FashionDataProcessor(
        mongo_uri="mongodb://localhost:27017",
        triton_url="localhost:8000", 
        image_folder="data/boys"
    )
    processor.process_excel("data/fashion.csv")