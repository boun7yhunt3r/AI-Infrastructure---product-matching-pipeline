# AI-Infrastructure---product-matching-pipeline
a product matching pipeline where an input image is compared against  stored products in a vector database and a MongoDB-based metadata store to find the  closest match.


# 🛍️ Product Matching Pipeline

A scalable system for matching product images against a database using visual and textual embeddings, powered by OpenAI's CLIP model. This pipeline integrates FAISS for similarity search, MongoDB for metadata storage, and NVIDIA Triton Inference Server for fast model serving — all wrapped with FastAPI and Docker for easy deployment.

## [Architecture]
<img width="3705" height="3705" alt="Untitled Diagram drawio" src="https://github.com/user-attachments/assets/37db5e1d-c4c1-4fd8-89f6-20a1dbe6e9aa" />

## ✨ Data Initialization
<img width="3855" height="1105" alt="flow drawio" src="https://github.com/user-attachments/assets/a1479d54-6ec9-4bda-b0aa-f9bcdcc33567" />

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

Clone the repository:

```bash
git clone https://github.com/your-username/product-matching-pipeline.git
cd product-matching-pipeline
