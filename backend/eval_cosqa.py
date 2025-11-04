from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Set

import uuid
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from sentence_transformers import SentenceTransformer
from qdrant_client.models import VectorParams, Distance

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

COSQA_COLLECTION = "cosqa_final_finetune"

def _collection_compatible(client: QdrantClient, name: str, dim_expected: int, dist_expected: str) -> bool:
    if not client.collection_exists(name):
        return False
    info = client.get_collection(name)
    vc = info.config.params.vectors  # VectorParams

    cur_size = int(vc.size)
    # Distance to enum, np. Distance.COSINE → value == 'Cosine'
    cur_dist = vc.distance.value if hasattr(vc.distance, "value") else str(vc.distance)
    cur_dist = cur_dist.upper()

    return cur_size == int(dim_expected) and cur_dist == dist_expected.upper()

def _collection_ready(client: QdrantClient, name: str, expected_count: int) -> bool:
    """
    Uznajemy kolekcję za gotową, jeśli:
    - istnieje,
    - ma poprawny wymiar i metrykę,
    - ma co najmniej expected_count punktów (dokładne liczenie).
    """
    if not _collection_compatible(client, name, settings.VECTOR_SIZE, settings.DISTANCE):
        return False
    try:
        count_info = client.count(collection_name=name, count_filter=None, exact=True)
        return int(count_info.count) >= int(expected_count)
    except Exception:
        return False

def ensure_cosqa_collection(client: QdrantClient, col_name):
    initialize_collection(client, collection_name=col_name)

def index_corpus(client, model, df_corpus, col_name,  batch_size: int = 256):
    ids = df_corpus["doc_id"].astype(str).tolist()
    texts = df_corpus["text"].astype(str).tolist()

    for start in range(0, len(texts), batch_size):
        end = min(start + batch_size, len(texts))
        chunk_ids = ids[start:end]
        chunk_txt = texts[start:end]

        # embed – zwróci dokładnie tyle wektorów, ile wejść
        vecs = model.encode(
            chunk_txt,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        # twarda weryfikacja zgodności długości – unikamy index out of range
        if len(vecs) != len(chunk_txt) or len(chunk_ids) != len(chunk_txt):
            raise RuntimeError(
                f"Batch size mismatch at [{start}:{end}]: "
                f"vecs={len(vecs)}, chunk_txt={len(chunk_txt)}, chunk_ids={len(chunk_ids)}"
            )

        # buduj punkty BEZ indeksowania po j — bezpiecznie po zip
        points = []
        for orig_id, txt, vec in zip(chunk_ids, chunk_txt, vecs):
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, orig_id))  # legalny UUID
            points.append(
                PointStruct(
                    id=point_id,
                    vector=vec.tolist(),
                    payload={
                        "doc_id": orig_id,           # używane w qrels i metrykach
                        "text": txt,
                    },
                )
            )

        client.upsert(collection_name=col_name, points=points, wait=True)


def retrieve_topk(client, model, queries, col_name, top_k: int = 10) -> Dict[str, List[str]]:
    out = {}
    for qid, qtext in queries:
        hits = search(
            client=client,
            query=qtext,
            model=model,
            top_k=top_k,
            collection_name=col_name,
        )
        # bierzemy doc_id z payloadu; jeśli z jakiegoś powodu go braknie – fallback do id
        out[qid] = [
            (h["payload"].get("doc_id") if h.get("payload") else None) or h["id"]
            for h in hits
        ]
    return out



def run_evaluation(
    split: str = "test",
    col_name: str = "cosqa",
    model_name: str = settings.MODEL_NAME, 
    limit_queries: Optional[int] = None,
    top_k: int = 10,
):

    client: QdrantClient = init_qdrant_client()
    model: SentenceTransformer = SentenceTransformer(model_name)
    print(model)

    df_corpus = load_corpus()                 # kolumny: doc_id, text
    df_queries = load_queries()               # kolumny: qid, query
    df_matches = load_matches(split)          # kolumny: query-id, corpus-id, score
    qrels = build_qrels(df_matches)  # qid -> {doc_id}

    used_qids = set(qrels.keys())
    queries = build_query_list(df_queries, used_qids)
    if limit_queries:
        queries = queries[:limit_queries]

    # # 2) Kolekcja + indeksacja (jednorazowo; w prostym wariancie robimy za każdym uruchomieniem)
    # ensure_cosqa_collection(client)
    # index_corpus(client, model, df_corpus, batch_size=512)

    expected_points = len(df_corpus)
    if _collection_ready(client, col_name, expected_points):
        # już gotowe – nic nie robimy
        print(f"Collection {col_name} ready")
        pass
    else:
        # utwórz/odtwórz zgodną kolekcję i zaindeksuj
        initialize_collection(client, collection_name=col_name)
        index_corpus(client, model, df_corpus, col_name, batch_size=512)

    # 3) Retrieval
    retrieved = retrieve_topk(client, model, queries, col_name, top_k=top_k)

    # 4) Metryki
    report = aggregate_at_k(retrieved, qrels, k=top_k)
    return report
