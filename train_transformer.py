import argparse
import json
import time
from collections import Counter
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

BASE_MODEL_ID = "cardiffnlp/twitter-roberta-base-sentiment-latest"
DATASET_PATH = Path("data/Twitter_Data.csv")
OUTPUT_DIR = Path("models/sentiment-transformer")
REPORTS_DIR = Path("reports")

METRICS_PATH = REPORTS_DIR / "fine_tuned_model_metrics.json"
SUMMARY_PATH = REPORTS_DIR / "fine_tuned_evaluation_summary.md"
CONFUSION_MATRIX_PATH = REPORTS_DIR / "fine_tuned_confusion_matrix.csv"

TEXT_COLUMN = "clean_text"
LABEL_COLUMN = "category"
LABEL_ORDER = ["Negative", "Neutral", "Positive"]

ID_TO_LABEL = {
    0: "Negative",
    1: "Neutral",
    2: "Positive",
}

LABEL_TO_ID = {
    "Negative": 0,
    "Neutral": 1,
    "Positive": 2,
}

# Dataset labels:
# -1 = Negative, 0 = Neutral, 1 = Positive
DATASET_LABEL_MAPPING = {
    -1: 0,
    0: 1,
    1: 2,
    "-1": 0,
    "0": 1,
    "1": 2,
}


class SentimentDataset(Dataset):
    """Small PyTorch dataset wrapper for tokenizer-based sentiment training."""

    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        encoding = self.tokenizer(
            self.texts[index],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[index], dtype=torch.long),
        }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune the SentiScope AI Studio transformer sentiment model."
    )
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional smaller stratified subset for testing on weak hardware.",
    )
    return parser.parse_args()


def normalize_dataset_label(value):
    if pd.isna(value):
        return None

    text_value = str(value).strip()
    numeric_value = None

    try:
        numeric_value = int(float(text_value))
    except ValueError:
        pass

    if numeric_value in DATASET_LABEL_MAPPING:
        return DATASET_LABEL_MAPPING[numeric_value]

    return DATASET_LABEL_MAPPING.get(text_value)


def load_and_clean_dataset():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}. Place Twitter_Data.csv inside data/."
        )

    dataframe = pd.read_csv(DATASET_PATH)

    missing_columns = [
        column for column in [TEXT_COLUMN, LABEL_COLUMN] if column not in dataframe.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    print("Raw class distribution:")
    print(dataframe[LABEL_COLUMN].value_counts(dropna=False).sort_index())

    dataframe = dataframe[[TEXT_COLUMN, LABEL_COLUMN]].copy()
    dataframe = dataframe.dropna(subset=[TEXT_COLUMN, LABEL_COLUMN])
    dataframe[TEXT_COLUMN] = dataframe[TEXT_COLUMN].astype(str).str.strip()
    dataframe = dataframe[dataframe[TEXT_COLUMN] != ""].copy()
    dataframe["label_id"] = dataframe[LABEL_COLUMN].apply(normalize_dataset_label)
    dataframe = dataframe[dataframe["label_id"].notna()].copy()
    dataframe["label_id"] = dataframe["label_id"].astype(int)

    if dataframe.empty:
        raise ValueError("No usable rows found after cleaning.")

    print("\nCleaned class distribution:")
    print(dataframe["label_id"].map(ID_TO_LABEL).value_counts().reindex(LABEL_ORDER, fill_value=0))

    return dataframe.reset_index(drop=True)


def stratified_subset(dataframe, max_rows, random_state):
    if max_rows is None or max_rows >= len(dataframe):
        return dataframe

    per_class = max_rows // len(LABEL_ORDER)
    if per_class == 0:
        raise ValueError("--max-rows must be at least 3 for stratified sampling.")

    parts = []
    for label_id in sorted(ID_TO_LABEL):
        rows = dataframe[dataframe["label_id"] == label_id]
        parts.append(rows.sample(n=min(per_class, len(rows)), random_state=random_state))

    sampled = pd.concat(parts).sample(frac=1, random_state=random_state).reset_index(drop=True)
    print(f"\nUsing stratified training subset: {len(sampled)} rows")
    return sampled


def stratified_split(dataframe, random_state):
    """Create train, validation, and test splits while preserving class balance."""
    train_parts = []
    validation_parts = []
    test_parts = []

    for label_id in sorted(ID_TO_LABEL):
        label_rows = dataframe[dataframe["label_id"] == label_id]
        label_rows = label_rows.sample(frac=1, random_state=random_state)

        train_end = int(len(label_rows) * 0.8)
        validation_end = int(len(label_rows) * 0.9)

        train_parts.append(label_rows.iloc[:train_end])
        validation_parts.append(label_rows.iloc[train_end:validation_end])
        test_parts.append(label_rows.iloc[validation_end:])

    train_data = pd.concat(train_parts).sample(frac=1, random_state=random_state).reset_index(drop=True)
    validation_data = pd.concat(validation_parts).sample(frac=1, random_state=random_state).reset_index(drop=True)
    test_data = pd.concat(test_parts).sample(frac=1, random_state=random_state).reset_index(drop=True)

    print("\nSplit sizes:")
    print(f"Train: {len(train_data)}")
    print(f"Validation: {len(validation_data)}")
    print(f"Test: {len(test_data)}")

    return train_data, validation_data, test_data


def build_dataloader(dataframe, tokenizer, max_length, batch_size, shuffle):
    dataset = SentimentDataset(
        texts=dataframe[TEXT_COLUMN].tolist(),
        labels=dataframe["label_id"].tolist(),
        tokenizer=tokenizer,
        max_length=max_length,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def build_confusion_matrix(true_labels, predicted_labels):
    matrix = {
        label: {predicted: 0 for predicted in LABEL_ORDER}
        for label in LABEL_ORDER
    }

    for true_label_id, predicted_label_id in zip(true_labels, predicted_labels):
        true_label = ID_TO_LABEL[int(true_label_id)]
        predicted_label = ID_TO_LABEL[int(predicted_label_id)]
        matrix[true_label][predicted_label] += 1

    return matrix


def compute_metrics(true_labels, predicted_labels):
    total = len(true_labels)
    correct = sum(1 for true, predicted in zip(true_labels, predicted_labels) if true == predicted)
    accuracy = correct / total if total else 0.0

    true_counts = Counter(true_labels)
    per_class = {}
    macro_precision = 0.0
    macro_recall = 0.0
    macro_f1 = 0.0

    for label_id, label_name in ID_TO_LABEL.items():
        true_positive = sum(
            1 for true, predicted in zip(true_labels, predicted_labels)
            if true == label_id and predicted == label_id
        )
        false_positive = sum(
            1 for true, predicted in zip(true_labels, predicted_labels)
            if true != label_id and predicted == label_id
        )
        false_negative = sum(
            1 for true, predicted in zip(true_labels, predicted_labels)
            if true == label_id and predicted != label_id
        )

        precision = (
            true_positive / (true_positive + false_positive)
            if (true_positive + false_positive)
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if (true_positive + false_negative)
            else 0.0
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )

        macro_precision += precision
        macro_recall += recall
        macro_f1 += f1

        per_class[label_name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": true_counts[label_id],
        }

    class_count = len(ID_TO_LABEL)
    return {
        "accuracy": accuracy,
        "macro_precision": macro_precision / class_count,
        "macro_recall": macro_recall / class_count,
        "macro_f1": macro_f1 / class_count,
        "per_class": per_class,
    }


def evaluate_model(model, dataloader, device):
    model.eval()
    true_labels = []
    predicted_labels = []
    total_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            total_loss += outputs.loss.item()
            predictions = torch.argmax(outputs.logits, dim=1)

            true_labels.extend(batch["labels"].detach().cpu().tolist())
            predicted_labels.extend(predictions.detach().cpu().tolist())

    metrics = compute_metrics(true_labels, predicted_labels)
    metrics["loss"] = total_loss / max(1, len(dataloader))
    metrics["confusion_matrix"] = build_confusion_matrix(true_labels, predicted_labels)
    return metrics


def train_one_epoch(model, dataloader, optimizer, scheduler, device, epoch_number):
    model.train()
    total_loss = 0.0

    for batch_index, batch in enumerate(dataloader, start=1):
        batch = {key: value.to(device) for key, value in batch.items()}

        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        if batch_index % 100 == 0 or batch_index == len(dataloader):
            print(
                f"Epoch {epoch_number} | Batch {batch_index}/{len(dataloader)} | "
                f"Loss: {loss.item():.4f}"
            )

    return total_loss / max(1, len(dataloader))


def confusion_matrix_to_csv(matrix):
    dataframe = pd.DataFrame(matrix).T
    dataframe = dataframe.reindex(index=LABEL_ORDER, columns=LABEL_ORDER, fill_value=0)
    dataframe.index.name = "Actual / Predicted"
    dataframe.to_csv(CONFUSION_MATRIX_PATH)


def save_reports(args, runtime_seconds, validation_history, test_metrics, train_size, validation_size, test_size):
    REPORTS_DIR.mkdir(exist_ok=True)
    confusion_matrix_to_csv(test_metrics["confusion_matrix"])

    output = {
        "base_model": BASE_MODEL_ID,
        "fine_tuned_model_path": str(OUTPUT_DIR),
        "dataset": str(DATASET_PATH),
        "label_mapping": {
            "-1": 0,
            "0": 1,
            "1": 2,
            "0_name": "Negative",
            "1_name": "Neutral",
            "2_name": "Positive",
        },
        "training_settings": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "max_length": args.max_length,
            "random_state": args.random_state,
            "max_rows": args.max_rows,
        },
        "split_sizes": {
            "train": train_size,
            "validation": validation_size,
            "test": test_size,
        },
        "validation_history": validation_history,
        "test_metrics": test_metrics,
        "training_runtime_seconds": runtime_seconds,
    }

    with METRICS_PATH.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=4)

    matrix = test_metrics["confusion_matrix"]
    summary = f"""# Fine-Tuned Transformer Evaluation Summary

## Base Model

`{BASE_MODEL_ID}`

## Fine-Tuned Model Path

`{OUTPUT_DIR}`

## Dataset

`{DATASET_PATH}`

## Label Mapping

```text
-1 -> 0 -> Negative
 0 -> 1 -> Neutral
 1 -> 2 -> Positive
```

## Training Settings

| Setting | Value |
| --- | ---: |
| Epochs | {args.epochs} |
| Batch Size | {args.batch_size} |
| Learning Rate | {args.learning_rate} |
| Weight Decay | {args.weight_decay} |
| Max Length | {args.max_length} |
| Random State | {args.random_state} |

## Split Sizes

| Split | Rows |
| --- | ---: |
| Train | {train_size} |
| Validation | {validation_size} |
| Test | {test_size} |

## Final Test Metrics

| Metric | Score |
| --- | ---: |
| Accuracy | {test_metrics['accuracy']:.4f} |
| Macro Precision | {test_metrics['macro_precision']:.4f} |
| Macro Recall | {test_metrics['macro_recall']:.4f} |
| Macro F1 | {test_metrics['macro_f1']:.4f} |

## Per-Class Test Metrics

| Class | Precision | Recall | F1 Score | Support |
| --- | ---: | ---: | ---: | ---: |
| Negative | {test_metrics['per_class']['Negative']['precision']:.4f} | {test_metrics['per_class']['Negative']['recall']:.4f} | {test_metrics['per_class']['Negative']['f1']:.4f} | {test_metrics['per_class']['Negative']['support']} |
| Neutral | {test_metrics['per_class']['Neutral']['precision']:.4f} | {test_metrics['per_class']['Neutral']['recall']:.4f} | {test_metrics['per_class']['Neutral']['f1']:.4f} | {test_metrics['per_class']['Neutral']['support']} |
| Positive | {test_metrics['per_class']['Positive']['precision']:.4f} | {test_metrics['per_class']['Positive']['recall']:.4f} | {test_metrics['per_class']['Positive']['f1']:.4f} | {test_metrics['per_class']['Positive']['support']} |

## Test Confusion Matrix

| Actual / Predicted | Negative | Neutral | Positive |
| --- | ---: | ---: | ---: |
| Negative | {matrix['Negative']['Negative']} | {matrix['Negative']['Neutral']} | {matrix['Negative']['Positive']} |
| Neutral | {matrix['Neutral']['Negative']} | {matrix['Neutral']['Neutral']} | {matrix['Neutral']['Positive']} |
| Positive | {matrix['Positive']['Negative']} | {matrix['Positive']['Neutral']} | {matrix['Positive']['Positive']} |

## Interpretation

The model was fine-tuned on the SentiScope AI Studio Twitter sentiment dataset using macro F1 as the main selection metric. Macro F1 gives equal importance to Negative, Neutral, and Positive classes, which is important because the original production baseline showed weak Positive recall.

The best checkpoint was selected using validation macro F1 and then evaluated on the held-out test split. The saved local model can now be used by the app through `model_service.py`.

Training runtime: **{runtime_seconds:.2f} seconds**
"""

    with SUMMARY_PATH.open("w", encoding="utf-8") as file:
        file.write(summary)


def main():
    args = parse_args()
    start_time = time.perf_counter()

    torch.manual_seed(args.random_state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training device: {device}")

    dataframe = load_and_clean_dataset()
    dataframe = stratified_subset(dataframe, args.max_rows, args.random_state)
    train_data, validation_data, test_data = stratified_split(dataframe, args.random_state)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL_ID,
        num_labels=3,
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
        ignore_mismatched_sizes=True,
    )
    model.to(device)

    train_loader = build_dataloader(train_data, tokenizer, args.max_length, args.batch_size, shuffle=True)
    validation_loader = build_dataloader(validation_data, tokenizer, args.max_length, args.batch_size, shuffle=False)
    test_loader = build_dataloader(test_data, tokenizer, args.max_length, args.batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    total_training_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_training_steps,
    )

    best_validation_f1 = -1.0
    validation_history = []

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device, epoch)
        validation_metrics = evaluate_model(model, validation_loader, device)
        validation_metrics["epoch"] = epoch
        validation_metrics["train_loss"] = train_loss
        validation_history.append(validation_metrics)

        print(
            f"Validation | Loss: {validation_metrics['loss']:.4f} | "
            f"Accuracy: {validation_metrics['accuracy']:.4f} | "
            f"Macro F1: {validation_metrics['macro_f1']:.4f}"
        )

        if validation_metrics["macro_f1"] > best_validation_f1:
            best_validation_f1 = validation_metrics["macro_f1"]
            model.save_pretrained(OUTPUT_DIR)
            tokenizer.save_pretrained(OUTPUT_DIR)
            print(f"Saved new best model to {OUTPUT_DIR}")

    print("\nLoading best checkpoint for test evaluation...")
    best_model = AutoModelForSequenceClassification.from_pretrained(OUTPUT_DIR)
    best_model.to(device)

    test_metrics = evaluate_model(best_model, test_loader, device)
    runtime_seconds = time.perf_counter() - start_time

    print("\nFinal Test Results")
    print(f"Accuracy:        {test_metrics['accuracy']:.4f}")
    print(f"Macro Precision: {test_metrics['macro_precision']:.4f}")
    print(f"Macro Recall:    {test_metrics['macro_recall']:.4f}")
    print(f"Macro F1:        {test_metrics['macro_f1']:.4f}")

    save_reports(
        args=args,
        runtime_seconds=runtime_seconds,
        validation_history=validation_history,
        test_metrics=test_metrics,
        train_size=len(train_data),
        validation_size=len(validation_data),
        test_size=len(test_data),
    )

    print(f"\nFine-tuned model saved to: {OUTPUT_DIR}")
    print(f"Metrics saved to: {METRICS_PATH}")
    print(f"Summary saved to: {SUMMARY_PATH}")
    print(f"Confusion matrix saved to: {CONFUSION_MATRIX_PATH}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print(f"Training stopped: {error}")
