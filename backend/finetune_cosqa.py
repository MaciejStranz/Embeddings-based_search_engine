from __future__ import annotations
import os
from typing import Optional, Tuple, List

import pandas as pd
from datasets import Dataset

from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    SentenceTransformerModelCardData,
)
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers

from .cosqa_adapter import load_corpus, load_queries, load_matches


def _build_pairs_df(split: str = "train", max_samples: Optional[int] = None) -> pd.DataFrame:
    """
    Buduje pary (query, code) dla MultipleNegativesRankingLoss z CoSQA:
      - bierzemy tylko score == 1 (pozytywne),
      - łączymy z treścią zapytań i dokumentów,
      - zwracamy DataFrame z kolumnami: sentence1 (query), sentence2 (code).
    """
    df_corpus = load_corpus().rename(columns={"doc_id": "corpus-id"})      # corpus-id, text
    df_queries = load_queries().rename(columns={"qid": "query-id"})         # query-id, query
    df_matches = load_matches(split)                                        # query-id, corpus-id, score

    df_pos = df_matches[df_matches["score"] == 1][["query-id", "corpus-id"]].copy()
    df_pos = df_pos.merge(df_queries, on="query-id", how="inner")
    df_pos = df_pos.merge(df_corpus, on="corpus-id", how="inner")

    df_out = pd.DataFrame({
        "sentence1": df_pos["query"].astype(str),  # anchor
        "sentence2": df_pos["text"].astype(str),   # positive
    })
    if max_samples is not None:
        df_out = df_out.iloc[:max_samples].reset_index(drop=True)
    return df_out


def _to_hf_dataset(df):
    # Zamiast Dataset.from_pandas(df, ...) — unikamy picklowania/dill
    return Dataset.from_dict({
        "sentence1": df["sentence1"].astype(str).tolist(),
        "sentence2": df["sentence2"].astype(str).tolist(),
    })


def finetune_with_trainer(
    base_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    output_dir: str = "models/cosqa-ft-trainer",
    train_split: str = "train",
    eval_split: Optional[str] = None,         # np. "valid" jeśli chcesz lightweight eval
    max_train_samples: Optional[int] = None,  # np. 20_000 na szybki run
    max_eval_samples: Optional[int] = None,   # np. 2_000
    num_train_epochs: int = 2,
    per_device_train_batch_size: int = 64,
    per_device_eval_batch_size: int = 64,
    learning_rate: float = 2e-5,
    warmup_ratio: float = 0.1,
    fp16: bool = True,
    bf16: bool = False,
    logging_steps: int = 100,
    save_steps: int = 1000,
    save_total_limit: int = 2,
    run_name: str = "cosqa-ft-mnlr",
) -> Tuple[str, Optional[str], Optional[str]]:
    """
    Trenuje model z MNLR na parach (query, code) w stylu SentenceTransformerTrainer.
    Zwraca ścieżki: (output_dir, log_json, final_model_dir).
    """
    # 1) Model + (opcjonalnie) dane do model card
    model = SentenceTransformer(
        base_model_name,
        model_card_data=SentenceTransformerModelCardData(
            language="en",
            license="apache-2.0",
            model_name=f"{base_model_name} fine-tuned on CoSQA (MNLR)",
        ),
    )

    # 2) Datasety (HF Dataset z kolumnami 'sentence1', 'sentence2')
    train_df = _build_pairs_df(split=train_split, max_samples=max_train_samples)
    train_ds = _to_hf_dataset(train_df)

    if eval_split:
        eval_df = _build_pairs_df(split=eval_split, max_samples=max_eval_samples)
        eval_ds = _to_hf_dataset(eval_df)
        eval_strategy = "steps"
    else:
        eval_ds = None
        eval_strategy = "no"

    # 3) Loss
    loss = MultipleNegativesRankingLoss(model)

    # 4) Argumenty treningowe (styl z przykładu)
    args = SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        fp16=fp16,
        bf16=bf16,
        batch_sampler=BatchSamplers.NO_DUPLICATES,   # MNLR lubi brak duplikatów w batchu
        eval_strategy=eval_strategy,
        eval_steps=logging_steps,                    # jeśli eval jest, trzymajmy jeden rytm
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        logging_steps=logging_steps,
        run_name=run_name,
    )

    # 5) Trainer
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        loss=loss,
        evaluator=None,   # można wpiąć własny evaluator retrieval; nie jest wymagany
    )

    # 6) Trening
    trainer.train()

    # 7) Zapisy
    final_dir = os.path.join(output_dir, "final")
    model.save_pretrained(final_dir)

    # 8) Logi: Hugging Face Trainer zapisuje log_history w trainer.state.log_history (list[dict])
    # Możesz wykorzystać to do wykresu loss vs. step (średnia per logging_steps).
    # Tu tylko zwrócimy ścieżki, a wykres można zrobić osobnym skryptem, jeśli chcesz.
    log_json = os.path.join(output_dir, "trainer_state.json") if os.path.exists(os.path.join(output_dir, "trainer_state.json")) else None

    return output_dir, log_json, final_dir
