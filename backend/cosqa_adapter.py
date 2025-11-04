from __future__ import annotations
from typing import Dict, Set, Tuple, List
import pandas as pd

SPLITS = {
    "train": "hf://datasets/CoIR-Retrieval/cosqa/data/train-00000-of-00001.parquet",
    "valid": "hf://datasets/CoIR-Retrieval/cosqa/data/valid-00000-of-00001.parquet",
    "test":  "hf://datasets/CoIR-Retrieval/cosqa/data/test-00000-of-00001.parquet",
}
CORPUS = "hf://datasets/CoIR-Retrieval/cosqa/corpus/corpus-00000-of-00001.parquet"
QUERIES = "hf://datasets/CoIR-Retrieval/cosqa/queries/queries-00000-of-00001.parquet"

def load_corpus() -> pd.DataFrame:
    df = pd.read_parquet(CORPUS)
    return df.rename(columns={"_id": "doc_id", "text": "text"})

def load_queries() -> pd.DataFrame:
    df = pd.read_parquet(QUERIES)
    return df.rename(columns={"_id": "qid", "text": "query"})

def load_matches(split: str = "test") -> pd.DataFrame:
    if split not in SPLITS:
        raise ValueError(f"Unknown split '{split}', choose from {list(SPLITS)}")
    df = pd.read_parquet(SPLITS[split])
    # columns: 'query-id', 'corpus-id', 'score'
    return df

def build_qrels(df_matches: pd.DataFrame) -> Dict[str, Set[str]]:
    df_pos = df_matches[df_matches["score"] == 1][["query-id", "corpus-id"]]
    grouped = df_pos.groupby("query-id")["corpus-id"].apply(lambda s: set(map(str, s.tolist())))
    return grouped.to_dict()

def build_query_list(df_queries: pd.DataFrame, used_qids: Set[str] | None = None) -> List[Tuple[str, str]]:
    dfq = df_queries[["qid", "query"]].copy()
    if used_qids is not None:
        dfq = dfq[dfq["qid"].astype(str).isin(used_qids)]
    dfq["qid"] = dfq["qid"].astype(str)
    dfq["query"] = dfq["query"].astype(str)
    return list(dfq.itertuples(index=False, name=None))  # (qid, query)
