import os
from pydantic import BaseModel, Field

class Settings(BaseModel):
    # Qdrant
    QDRANT_HOST: str = Field(default=os.getenv("QDRANT_HOST", "localhost"))
    QDRANT_PORT: int = Field(default=int(os.getenv("QDRANT_PORT", "6333")))
    COLLECTION_NAME: str = Field(default=os.getenv("COLLECTION_NAME", "docs"))

    # Embeddings
    MODEL_NAME: str = Field(default=os.getenv("MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"))
    VECTOR_SIZE: int = Field(default=int(os.getenv("VECTOR_SIZE", "384")))
    DISTANCE: str = Field(default=os.getenv("DISTANCE", "COSINE").upper())  

    # Accepting a collection of documents on start
    CSV_PATH: str | None = Field(default=os.getenv("CSV_PATH", "movies_reviews.csv"))  
    

settings = Settings()
