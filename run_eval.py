from backend.eval_cosqa import run_evaluation

if __name__ == "__main__":
    # Zmień split na "train"/"valid"/"test" według potrzeb
    report = run_evaluation(split="test", limit_queries=None, top_k=10)
    print("=== CoSQA Evaluation (split=test, k=10) ===")
    for k, v in report.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
