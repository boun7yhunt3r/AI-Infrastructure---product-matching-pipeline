Product Matching Pipeline
https://i.imgur.com/JKQ1x8l.png

A scalable system for matching product images against a database using visual and textual embeddings powered by CLIP model, FAISS for similarity search, and MongoDB for metadata storage.

Features
Visual Language Model: CLIP model for image and text embeddings

Vector Database: FAISS for efficient nearest neighbor search

Metadata Storage: MongoDB for product information

Model Serving: NVIDIA Triton Inference Server with TensorRT optimization

REST API: FastAPI for easy integration

Logging: Comprehensive logging to MongoDB

Containerized: Docker for easy deployment

Architecture
The system consists of three main components:

Vector Database (FAISS): Stores product embeddings for efficient similarity search

Metadata Database (MongoDB): Stores product information (name, category, price, etc.)

Inference Service (Triton): Hosts the quantized CLIP model for embedding generation

Getting Started
Prerequisites
Docker and Docker Compose

NVIDIA GPU with drivers (for TensorRT acceleration)

Python 3.9+

Installation
Clone the repository:

bash
git clone https://github.com/your-username/product-matching-pipeline.git
cd product-matching-pipeline
Build and start the services:

bash
docker-compose up --build
Initialize the database with sample data:

bash
docker exec -it product-matcher-app-1 python -c "
from database.mongo_db import MongoDB
import json

db = MongoDB('mongodb://mongodb:27017')
with open('data/sample_products.json') as f:
    products = json.load(f)
    for product in products:
        db.insert_product(product)
print('Database initialized with sample data')
"
API Documentation
Endpoints
POST /match-product: Match an input image against stored products

Input: Image file (JPEG/PNG)

Output: Best matching product with metadata and similarity score

GET /health: Service health check

Example Request
bash
curl -X POST -F "image=@data/sample_images/test_shoe.jpg" http://localhost:8000/match-product
Example Response
json
{
  "status": "success",
  "result": {
    "match": {
      "_id": "1",
      "name": "Nike Air Max Running Shoes",
      "category": "Footwear",
      "price": 129.99,
      "description": "Comfortable running shoes with air cushioning"
    },
    "distance": 0.15
  }
}
Configuration
Environment variables can be set in the .env file:

text
MONGODB_URI=mongodb://mongodb:27017
TRITON_URL=triton:8001
Directory Structure
text
product-matcher/
├── app/                  # Application code
│   ├── database/        # Database connectors
│   ├── models/          # Model implementations
│   ├── services/        # Business logic
│   ├── utils/           # Utility functions
│   └── triton/          # Triton configuration
├── data/                # Sample data
├── tests/               # Unit tests
├── Dockerfile           # Application Dockerfile
├── docker-compose.yml   # Service orchestration
└── requirements.txt     # Python dependencies
Performance Optimization
Model Quantization: CLIP model quantized to FP16/INT8 using TensorRT

Batch Processing: Triton configured for batch inference (max_batch_size=8)

Efficient Search: FAISS for fast nearest neighbor search

Async Processing: FastAPI async endpoints for concurrent requests

Extending the System
Add support for additional Visual Language Models

Implement hybrid search combining text and image embeddings

Add user feedback mechanism to improve matching

Scale horizontally with Kubernetes

Troubleshooting
Issue: Triton server fails to start

Solution: Ensure you have NVIDIA GPU drivers installed and nvidia-docker configured

Issue: MongoDB connection errors

Solution: Check if MongoDB container is running and accessible at the configured URI

Issue: FAISS index not working

Solution: Verify the embedding dimensions match between the model and FAISS index

License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgments
OpenAI for the CLIP model

Facebook Research for FAISS

NVIDIA for Triton Inference Server