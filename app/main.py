from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from database.mongo_db import MongoDB
from database.vector_db import VectorDB
from services.matching import ProductMatcher
from utils.logging import setup_logging
import os

app = FastAPI(title="Product Matching Service")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


mongodb_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
triton_url = os.getenv("TRITON_URL", "localhost:8001")

mongo_db = MongoDB(mongodb_uri)
vector_db = VectorDB()
logger = setup_logging(mongo_db)
matcher = ProductMatcher(mongo_db, vector_db, triton_url, logger)

@app.post("/match-product")
async def match_product(image: UploadFile = File(...)):
    try:
        result = await matcher.match_product(image)
        return {"status": "success", "result": result}
    except Exception as e:
        logger.log_error(str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "healthy"}