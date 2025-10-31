import os
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import pandas as pd
from sentence_transformers import SentenceTransformer
from .config import settings


def init_qdrant_client() -> QdrantClient:
    return QdrantClient(host = settings.QDRANT_HOST, port = settings.QDRANT_PORT)

def init_model() -> SentenceTransformer:
    return SentenceTransformer(settings.MODEL_NAME)

def initialize_collection(client: QdrantClient, collection_name: str = settings.COLLECTION_NAME):
    if not client.collection_exists(collection_name):
        client.recreate_collection(
            collection_name = collection_name,
            vectors_config = VectorParams(
                size=settings.VECTOR_SIZE,
                distance=getattr(Distance, settings.DISTANCE)
            ),
        )
        return True
    return False

    
def insert_data_from_csv(client: QdrantClient, model: SentenceTransformer, collection_name: str = settings.COLLECTION_NAME, path: str = settings.CSV_PATH):
    texts = pd.read_csv(path, header=None)[0].astype(str).tolist()
    texts = texts[0:50]
    embeddings = model.encode(texts, normalize_embeddings=True)
    client.upsert(
    collection_name=collection_name,
    points=[
        PointStruct(
                id = i,
                vector = vector.tolist(),
                payload = {"review": text}
        )
        for i, (text, vector) in enumerate(zip(texts, embeddings))
    ]
    )
    
def search(client: QdrantClient, query: str, model: SentenceTransformer, collection_name: str = settings.COLLECTION_NAME):
    vquery = model.encode(query, normalize_embeddings=True)
    search_result = client.query_points(
        collection_name=collection_name,
        query=vquery,
        with_payload=True,
        limit=5
    ).points
    results = [
        {"id": str(r.id), "score": float(r.score), "payload": r.payload}
        for r in search_result
    ] 
    return results
    
def insert_doc(client :QdrantClient, model: SentenceTransformer, text: str, collection_name: str = settings.COLLECTION_NAME):
    pass




