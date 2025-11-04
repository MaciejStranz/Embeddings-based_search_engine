# schemas.py
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, Union, List

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Query text for semantic search")
    top_k: int = Field(10, description="Number of top results to return")
    collection: Optional[str] = Field(
        None,
        description="Optional name of Qdrant collection to search. Defaults to the configured collection if not provided.",
    )

class SearchHit(BaseModel):
    id: str = Field(..., description="Qdrant point ID")
    score: float = Field(..., description="Similarity score")
    payload: Dict[str, Any] = Field(..., description="Stored metadata or text")

class SearchResponse(BaseModel):
    query: str = Field(..., description="Original query text")
    results: List[SearchHit] = Field(..., description="List of retrieved documents")

class AddDocRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text or document to be indexed")
    collection: Optional[str] = Field(
        None,
        description="Target Qdrant collection name. If not provided, defaults to the configured collection.",
    )

class AddDocResponse(BaseModel):
    id: str = Field(..., description="Stored document ID in Qdrant (UUIDv4)")
    collection: str = Field(..., description="Collection where the document was stored")
    payload: Dict[str, Any] = Field(..., description="Document metadata")
