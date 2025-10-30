import os
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import pandas as pd
from sentence_transformers import SentenceTransformer

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "docs")
VECTOR_SIZE = int(os.getenv("VECTOR_SIZE", "384")) #384 size of embedding from model, 4 just for tests
DISTANCE = os.getenv("VECTOR_DISTANCE", "COSINE")  # Cosine | Dot | Euclid


def get_client() -> QdrantClient:
    return QdrantClient(host = QDRANT_HOST, port = QDRANT_PORT)

def initialize_collection(client: QdrantClient, collection_name: str = COLLECTION_NAME):
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name = collection_name,
            vectors_config = VectorParams(
                size=VECTOR_SIZE,
                distance=getattr(Distance, DISTANCE)
            ),
        )
        return True
    return False

    

def insert_data_from_csv(client: QdrantClient, model: SentenceTransformer, collection_name: str = COLLECTION_NAME, path: str = "../movies_reviews.csv"):
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
    
def search(client: QdrantClient, query: str, model: SentenceTransformer, collection_name: str = COLLECTION_NAME):
    vquery = model.encode(query, normalize_embeddings=True)
    search_result = client.query_points(
        collection_name=collection_name,
        query=vquery,
        with_payload=True,
        limit=5
    ).points
    return search_result
    

# def insert_test_data(client: QdrantClient, collection_name: str = COLLECTION_NAME):
#     operation_info = client.upsert(
#         collection_name=collection_name,
#         wait=True,
#         points=[
#             PointStruct(id=1, vector=[0.05, 0.61, 0.76, 0.74], payload={"city": "Berlin"}),
#             PointStruct(id=2, vector=[0.19, 0.81, 0.75, 0.11], payload={"city": "London"}),
#             PointStruct(id=3, vector=[0.36, 0.55, 0.47, 0.94], payload={"city": "Moscow"}),
#             PointStruct(id=4, vector=[0.18, 0.01, 0.85, 0.80], payload={"city": "New York"}),
#             PointStruct(id=5, vector=[0.24, 0.18, 0.22, 0.44], payload={"city": "Beijing"}),
#             PointStruct(id=6, vector=[0.35, 0.08, 0.11, 0.44], payload={"city": "Mumbai"}),
#         ],
#     )
#     print(operation_info)



