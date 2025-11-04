from __future__ import annotations
from typing import List, Dict, Set
import math

def recall_at_k(retrieved: List[str], relevant: Set[str], k: int = 10) -> float:
    if not relevant:
        return 0.0
    topk = retrieved[:k]
    hits = sum(1 for d in topk if d in relevant)
    return hits / len(relevant)

def rr_at_k(retrieved: List[str], relevant: Set[str], k: int = 10) -> float:
    topk = retrieved[:k]
    for i, d in enumerate(topk, start=1):
        if d in relevant:
            return 1.0 / i
    return 0.0

def _dcg_at_k(binary_rels: List[int], k: int = 10) -> float:
    dcg = 0.0
    for i, rel in enumerate(binary_rels[:k], start=1):
        if rel > 0:
            dcg += (2**rel - 1) / math.log2(i + 1)
    return dcg

def ndcg_at_k(retrieved: List[str], relevant: Set[str], k: int = 10) -> float:
    rels = [1 if d in relevant else 0 for d in retrieved[:k]]
    dcg = _dcg_at_k(rels, k)

    ideal_ones = min(k, len(relevant))
    ideal_rels = [1] * ideal_ones + [0] * (k - ideal_ones)
    idcg = _dcg_at_k(ideal_rels, k)
    return (dcg / idcg) if idcg > 0 else 0.0


def aggregate_at_k(all_retrieved: Dict[str, List[str]], qrels: Dict[str, Set[str]], k: int = 10):
    recalls, rrs, ndcgs = [], [], []
    evaluated = 0
    for qid, retrieved in all_retrieved.items():
        relevant = qrels.get(qid, set())
        if not relevant:
            continue  
        recalls.append(recall_at_k(retrieved, relevant, k))
        rrs.append(rr_at_k(retrieved, relevant, k))
        ndcgs.append(ndcg_at_k(retrieved, relevant, k))
        evaluated += 1
    avg = lambda xs: sum(xs) / len(xs) if xs else 0.0
    return {
        "queries_evaluated": evaluated,
        f"Recall@{k}": avg(recalls),
        f"MRR@{k}": avg(rrs),
        f"nDCG@{k}": avg(ndcgs),
    }
