from __future__ import annotations
from typing import Dict, Set, Tuple, List
import pandas as pd

# Domyślne ścieżki Parquet na HF Hub (działają z Pandas 2.2+)
SPLITS = {
    "train": "hf://datasets/CoIR-Retrieval/cosqa/data/train-00000-of-00001.parquet",
    "valid": "hf://datasets/CoIR-Retrieval/cosqa/data/valid-00000-of-00001.parquet",
    "test":  "hf://datasets/CoIR-Retrieval/cosqa/data/test-00000-of-00001.parquet",
}
CORPUS = "hf://datasets/CoIR-Retrieval/cosqa/corpus/corpus-00000-of-00001.parquet"
QUERIES = "hf://datasets/CoIR-Retrieval/cosqa/queries/queries-00000-of-00001.parquet"

def load_corpus() -> pd.DataFrame:
    df = pd.read_parquet(CORPUS)
    # oczekiwane kolumny: '_id', 'text' (i ewentualne meta)
    return df.rename(columns={"_id": "doc_id", "text": "text"})

def load_queries() -> pd.DataFrame:
    df = pd.read_parquet(QUERIES)
    # oczekiwane kolumny: '_id', 'text'
    return df.rename(columns={"_id": "qid", "text": "query"})

def load_matches(split: str = "test") -> pd.DataFrame:
    if split not in SPLITS:
        raise ValueError(f"Unknown split '{split}', choose from {list(SPLITS)}")
    df = pd.read_parquet(SPLITS[split])
    # kolumny: 'query-id', 'corpus-id', 'score'
    return df

def build_qrels(df_matches: pd.DataFrame) -> Dict[str, Set[str]]:
    # bierzemy wyłącznie pozytywne pary (score == 1)
    df_pos = df_matches[df_matches["score"] == 1][["query-id", "corpus-id"]]
    # qid -> set(doc_id) (w tym datasie typowo 1 element)
    grouped = df_pos.groupby("query-id")["corpus-id"].apply(lambda s: set(map(str, s.tolist())))
    return grouped.to_dict()

def build_query_list(df_queries: pd.DataFrame, used_qids: Set[str] | None = None) -> List[Tuple[str, str]]:
    dfq = df_queries[["qid", "query"]].copy()
    if used_qids is not None:
        dfq = dfq[dfq["qid"].astype(str).isin(used_qids)]
    dfq["qid"] = dfq["qid"].astype(str)
    dfq["query"] = dfq["query"].astype(str)
    return list(dfq.itertuples(index=False, name=None))  # (qid, query)
