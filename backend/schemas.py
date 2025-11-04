from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional

# Request/response do API wyszukiwania
class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=10, ge=1, le=100)
    collection_name: Optional[str] = Field(None)

class SearchHit(BaseModel):
    id: str
    score: float
    payload: Dict[str, Any] | None = None

class SearchResponse(BaseModel):
    query: str
    results: List[SearchHit]

# For further development - adding doc through api
class Document(BaseModel):
    id: str | int
    text: str
    metadata: Dict[str, Any] | None = None

class IngestRequest(BaseModel):
    collection: str | None = None
    documents: List[Document]
    upsert: bool = True
