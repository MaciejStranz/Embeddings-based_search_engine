import json
import pandas as pd
import matplotlib.pyplot as plt
import os

STATE_PATH = "models2/cosqa-ft-trainer/checkpoint-141/trainer_state.json"   # path to file with final logs
OUTPUT_FIG = "loss_curve.png"     

def main():
    if not os.path.exists(STATE_PATH):
        raise FileNotFoundError(f"File not found: {STATE_PATH}")

    with open(STATE_PATH, "r", encoding="utf-8") as f:
        state = json.load(f)

    log_history = pd.DataFrame(state.get("log_history", []))

    train_loss = log_history[["step", "loss"]].dropna()
    eval_loss = log_history[["step", "eval_loss"]].dropna()

    if train_loss.empty and eval_loss.empty:
        print("No loss data found in trainer_state.json")
        return

    plt.figure(figsize=(10, 6))
    if not train_loss.empty:
        plt.plot(train_loss["step"], train_loss["loss"], label="Train Loss", marker="o")
    if not eval_loss.empty:
        plt.plot(eval_loss["step"], eval_loss["eval_loss"], label="Eval Loss", marker="s")

    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.title("Training and Evaluation Loss over Steps")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(OUTPUT_FIG)
    print(f"Saved plot to {OUTPUT_FIG}")
    plt.show()

if __name__ == "__main__":
    main()