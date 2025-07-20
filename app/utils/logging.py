from typing import Dict, Any
from database.mongo_db import MongoDB

class Logger:
    def __init__(self, mongo_db: MongoDB):
        self.mongo_db = mongo_db
    
    def log_event(self, data: Dict[str, Any]) -> str:
        """Log an event to MongoDB"""
        return self.mongo_db.log_event(data)
    
    def log_error(self, error: str) -> str:
        """Log an error to MongoDB"""
        return self.log_event({
            "type": "error",
            "error": error,
            "timestamp": datetime.utcnow()
        })

def setup_logging(mongo_db: MongoDB) -> Logger:
    return Logger(mongo_db)