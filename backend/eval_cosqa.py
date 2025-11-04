from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Set

import uuid
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from sentence_transformers import SentenceTransformer

from .config import settings
from .search_engine import (
    init_qdrant_client,
    init_model,
    initialize_collection,
    search,  
)
from .metrics import aggregate_at_k
from .cosqa_adapter import (
    load_corpus,
    load_queries,
    load_matches,
    build_qrels,
    build_query_list,
)

COSQA_COLLECTION = "cosqa_eval"

def ensure_cosqa_collection(client: QdrantClient):
    initialize_collection(client, collection_name=COSQA_COLLECTION)

def index_corpus(client, model, df_corpus, batch_size: int = 512):
    ids = df_corpus["doc_id"].astype(str).tolist()
    texts = df_corpus["text"].astype(str).tolist()

    for i in range(0, len(texts), batch_size):
        chunk_ids = ids[i : i + batch_size]
        chunk_txt = texts[i : i + batch_size]
        vecs = model.encode(chunk_txt, normalize_embeddings=True)

        points = []
        for j in range(len(chunk_txt)):
            original_doc_id = chunk_ids[j]                 # np. "d123"
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, original_doc_id))  # valid UUID
            points.append(
                PointStruct(
                    id=point_id,
                    vector=vecs[j].tolist(),
                    payload={
                        "doc_id": original_doc_id,         # ← oryginalny corpus-id
                        "snippet": chunk_txt[j][:512],
                    },
                )
            )

        client.upsert(collection_name=COSQA_COLLECTION, points=points, wait=True)


def retrieve_topk(client, model, queries, top_k: int = 10) -> Dict[str, List[str]]:
    out = {}
    for qid, qtext in queries:
        hits = search(
            client=client,
            query=qtext,
            model=model,
            top_k=top_k,
            collection_name=COSQA_COLLECTION,
        )
        # bierzemy doc_id z payloadu; jeśli z jakiegoś powodu go braknie – fallback do id
        out[qid] = [
            (h["payload"].get("doc_id") if h.get("payload") else None) or h["id"]
            for h in hits
        ]
    return out



def run_evaluation(
    split: str = "test",
    limit_queries: Optional[int] = None,
    top_k: int = 10,
):

    client: QdrantClient = init_qdrant_client()
    model: SentenceTransformer = init_model()

    df_corpus = load_corpus()                 # kolumny: doc_id, text
    df_queries = load_queries()               # kolumny: qid, query
    df_matches = load_matches(split)          # kolumny: query-id, corpus-id, score
    qrels: Dict[str, Set[str]] = build_qrels(df_matches)  # qid -> {doc_id}

    used_qids: Set[str] = set(qrels.keys())
    queries: List[Tuple[str, str]] = build_query_list(df_queries, used_qids)
    if limit_queries:
        queries = queries[:limit_queries]

    # 2) Kolekcja + indeksacja (jednorazowo; w prostym wariancie robimy za każdym uruchomieniem)
    ensure_cosqa_collection(client)
    index_corpus(client, model, df_corpus, batch_size=512)

    # 3) Retrieval
    retrieved = retrieve_topk(client, model, queries, top_k=top_k)

    # 4) Metryki
    report = aggregate_at_k(retrieved, qrels, k=top_k)
    return report
