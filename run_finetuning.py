from backend.finetune_cosqa import finetune_with_trainer

if __name__ == "__main__":
    out_dir, log_json, final_dir = finetune_with_trainer(
        base_model_name="sentence-transformers/all-MiniLM-L6-v2",
        output_dir="models2/cosqa-ft-trainer",
        train_split="train",
        eval_split="valid",           
        max_train_samples=10000,   
        max_eval_samples=None,
        num_train_epochs=1, 
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        fp16=True,
        bf16=False,
        logging_steps=5,
        save_steps=10,
        save_total_limit=None,
        run_name="cosqa-ft-mnlr2",
    )

    print(f"[OK] Output dir:      {out_dir}")
    print(f"[OK] Final model dir: {final_dir}")

