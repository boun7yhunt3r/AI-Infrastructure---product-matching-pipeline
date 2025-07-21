# AI-Infrastructure---product-matching-pipeline
a product matching pipeline where an input image is compared against  stored products in a vector database and a MongoDB-based metadata store to find the  closest match.


# 🛍️ Product Matching Pipeline

A scalable system for matching product images against a database using visual and textual embeddings, powered by OpenAI's CLIP model. This pipeline integrates FAISS for similarity search, MongoDB for metadata storage, and NVIDIA Triton Inference Server for fast model serving — all wrapped with FastAPI and Docker for easy deployment.

## [Architecture]
<img width="3705" height="3705" alt="Untitled Diagram drawio" src="https://github.com/user-attachments/assets/37db5e1d-c4c1-4fd8-89f6-20a1dbe6e9aa" />

## ✨ Data Initialization
<img width="4155" height="1405" alt="Untitled Diagram drawio (1)" src="https://github.com/user-attachments/assets/32408af1-18a9-4efb-bc91-6d6ec632bb30" />

## ✨ Query
<img width="5555" height="605" alt="Untitled Diagram drawio (2)" src="https://github.com/user-attachments/assets/d6b1de58-f448-43cb-980f-8fb538123f67" />


---

## ✨ Features

- **Visual Language Model**: CLIP model for generating image and text embeddings
- **Vector Database**: FAISS for efficient nearest neighbor search
- **Metadata Storage**: MongoDB for storing product information
- **Model Serving**: NVIDIA Triton Inference Server with TensorRT optimization
- **REST API**: FastAPI for simple and fast integration
- **Logging**: Comprehensive logging to MongoDB
- **Containerized**: Fully Dockerized for consistent deployments

---

## 🧱 Architecture Overview

The system consists of three main components:

1. **FAISS Vector Store**: Stores product embeddings for fast similarity search
2. **MongoDB Metadata Store**: Stores detailed product information (name, category, price, etc.)
3. **Triton Inference Server**: Hosts the quantized CLIP model for fast embedding generation

---

## 🚀 Getting Started

### ✅ Prerequisites

- Docker & Docker Compose
- NVIDIA GPU with drivers and `nvidia-docker2` installed
- Python 3.9+

### 📦 Installation

Build and start the services:

bash
docker-compose up --build

Initialize the MongoDB with sample product data:
Data/

📡 API Documentation
🔍 POST /match-product
Match an input image against stored products.

Input: JPEG or PNG image

Output: Best matching product with similarity score

🧪 Example Request
curl -X POST -F "test_shoe.jpg" http://localhost:8000/match-product

✅ Example Response
json
Copy
Edit
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

🧪 GET /health
Check if the API service is alive.

⚙️ Configuration
Environment variables are managed in the .env file:

env
MONGODB_URI=mongodb://mongodb:27017
TRITON_URL=triton:8001
