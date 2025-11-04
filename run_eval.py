from backend.eval_cosqa import run_evaluation

if __name__ == "__main__":
    path = "models2/cosqa-ft-trainer/final"
    col_name = "cosqa_final_finetuning"
    report = run_evaluation(split="test", col_name = col_name, model_name=path, limit_queries=None, top_k=10)
    print("=== CoSQA Evaluation (split=test, k=10) ===")
    for k, v in report.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
