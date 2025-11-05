import argparse
import re
from backend.eval_cosqa import run_evaluation

def _slugify(s: str, max_len: int = 60) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9_-]+", "_", s)
    s = s.strip("_")
    return (s[:max_len] if len(s) > max_len else s) or "model"

def parse_args():
    p = argparse.ArgumentParser(
        description="Run CoSQA evaluation for a chosen model and collection."
    )
    p.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Path or HF hub name of the SentenceTransformer model "
             "(e.g., 'sentence-transformers/all-MiniLM-L6-v2' or 'models2/cosqa-ft-trainer/final').",
    )
    p.add_argument(
        "--collection",
        default="cosqa",
        help="Qdrant collection name. If omitted, it will be derived from the model path: slug(model) + '_cosqa'.",
    )
    p.add_argument(
        "--k",
        type=int,
        default=10,
        help="Top-K for metrics (Recall@K, MRR@K, nDCG@K). Default: 10",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit of queries to evaluate (for quick runs).",
    )
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # Derive collection name if not provided
    col_name = args.collection or f"{_slugify(args.model)}_cosqa"

    report = run_evaluation(
        split="test",
        col_name=col_name,
        model_name=args.model,
        limit_queries=args.limit,
        top_k=args.k,
    )

    print(f"=== CoSQA Evaluation (split=test k={args.k}) ===")
    for k, v in report.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
