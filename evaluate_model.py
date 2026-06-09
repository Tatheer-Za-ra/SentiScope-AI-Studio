import argparse
import json
import time
from collections import Counter
from pathlib import Path

import pandas as pd

from model_service import MODEL_ID, predict_sentiment

DATASET_PATH = Path("data/Twitter_Data.csv")
REPORTS_DIR = Path("reports")
METRICS_PATH = REPORTS_DIR / "model_metrics.json"
SUMMARY_PATH = REPORTS_DIR / "evaluation_summary.md"
CONFUSION_MATRIX_PATH = REPORTS_DIR / "confusion_matrix.csv"

TEXT_COLUMN = "clean_text"
LABEL_COLUMN = "category"
DEFAULT_SAMPLE_SIZE = 500
DEFAULT_RANDOM_STATE = 42

LABEL_ORDER = ["Negative", "Neutral", "Positive"]

# Dataset label mapping for data/Twitter_Data.csv:
# -1 = Negative, 0 = Neutral, 1 = Positive
DATASET_LABEL_MAPPING = {
    -1: "Negative",
    0: "Neutral",
    1: "Positive",
    "-1": "Negative",
    "0": "Neutral",
    "1": "Positive",
    "negative": "Negative",
    "neutral": "Neutral",
    "positive": "Positive",
}

# Production model output mapping:
# cardiffnlp/twitter-roberta-base-sentiment-latest uses
# 0 = negative, 1 = neutral, 2 = positive internally.
MODEL_LABEL_MAPPING = {
    0: "Negative",
    1: "Neutral",
    2: "Positive",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate the SentiScope AI Studio production sentiment model."
    )
    parser.add_argument(
        "--sample-size",
        default=str(DEFAULT_SAMPLE_SIZE),
        help="Number of rows to evaluate, or 'full' to evaluate all cleaned rows.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help="Random seed used for repeatable sampling.",
    )
    parser.add_argument(
        "--balanced",
        action="store_true",
        help="Sample equal records from Negative, Neutral, and Positive classes.",
    )
    return parser.parse_args()


def normalize_sample_size(value):
    if str(value).strip().lower() == "full":
        return "full"

    try:
        sample_size = int(value)
    except ValueError as error:
        raise ValueError("--sample-size must be a positive integer or 'full'.") from error

    if sample_size <= 0:
        raise ValueError("--sample-size must be greater than 0.")

    return sample_size


def normalize_label(value):
    """Convert dataset labels into Negative, Neutral, or Positive."""
    if pd.isna(value):
        return None

    text_value = str(value).strip().lower()
    numeric_value = None

    try:
        numeric_value = int(float(text_value))
    except ValueError:
        pass

    if numeric_value in DATASET_LABEL_MAPPING:
        return DATASET_LABEL_MAPPING[numeric_value]

    return DATASET_LABEL_MAPPING.get(text_value)


def format_distribution(series):
    return series.value_counts(dropna=False).reindex(LABEL_ORDER, fill_value=0)


def load_and_clean_dataset():
    """Load the dataset and clean missing text/category values."""
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}. Place Twitter_Data.csv inside the data folder."
        )

    dataframe = pd.read_csv(DATASET_PATH)

    missing_columns = [
        column for column in [TEXT_COLUMN, LABEL_COLUMN] if column not in dataframe.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}. Expected {TEXT_COLUMN} and {LABEL_COLUMN}."
        )

    print("\nClass distribution before cleaning:")
    print(dataframe[LABEL_COLUMN].value_counts(dropna=False).sort_index())

    dataframe = dataframe[[TEXT_COLUMN, LABEL_COLUMN]].copy()
    dataframe = dataframe.dropna(subset=[TEXT_COLUMN, LABEL_COLUMN])
    dataframe[TEXT_COLUMN] = dataframe[TEXT_COLUMN].astype(str).str.strip()
    dataframe = dataframe[dataframe[TEXT_COLUMN] != ""].copy()
    dataframe["true_label"] = dataframe[LABEL_COLUMN].apply(normalize_label)

    print("\nFirst 20 labels before and after conversion:")
    print(dataframe[[LABEL_COLUMN, "true_label"]].head(20).to_string(index=False))

    dataframe = dataframe[dataframe["true_label"].notna()].copy()

    if dataframe.empty:
        raise ValueError(
            "No usable rows found after cleaning text and labels. Check DATASET_LABEL_MAPPING."
        )

    print("\nClass distribution after cleaning and label mapping:")
    print(format_distribution(dataframe["true_label"]))

    return dataframe.reset_index(drop=True)


def balanced_sample(dataframe, sample_size, random_state):
    """Sample an equal number of records from each class."""
    available_counts = dataframe["true_label"].value_counts()
    min_available = min(available_counts.get(label, 0) for label in LABEL_ORDER)

    if min_available == 0:
        raise ValueError("Balanced sampling requires at least one row in every class.")

    if sample_size == "full":
        per_class = min_available
    else:
        per_class = min(sample_size // len(LABEL_ORDER), min_available)

    if per_class == 0:
        raise ValueError(
            f"Balanced sampling needs at least {len(LABEL_ORDER)} rows. Increase --sample-size."
        )

    sampled_parts = []
    for label in LABEL_ORDER:
        label_rows = dataframe[dataframe["true_label"] == label]
        sampled_parts.append(
            label_rows.sample(n=per_class, random_state=random_state)
        )

    return pd.concat(sampled_parts).sample(frac=1, random_state=random_state).reset_index(drop=True)


def regular_sample(dataframe, sample_size, random_state):
    """Sample randomly while preserving the natural dataset distribution."""
    if sample_size == "full" or sample_size >= len(dataframe):
        return dataframe.sample(frac=1, random_state=random_state).reset_index(drop=True)

    return dataframe.sample(n=sample_size, random_state=random_state).reset_index(drop=True)


def sample_dataset(dataframe, sample_size, random_state, balanced):
    if balanced:
        sampled = balanced_sample(dataframe, sample_size, random_state)
    else:
        sampled = regular_sample(dataframe, sample_size, random_state)

    print("\nClass distribution after sampling:")
    print(format_distribution(sampled["true_label"]))
    return sampled


def build_confusion_matrix(true_labels, predicted_labels):
    matrix = {
        true_label: {predicted_label: 0 for predicted_label in LABEL_ORDER}
        for true_label in LABEL_ORDER
    }

    for true_label, predicted_label in zip(true_labels, predicted_labels):
        matrix[true_label][predicted_label] += 1

    return matrix


def compute_metrics(true_labels, predicted_labels):
    """Compute accuracy and weighted precision/recall/F1 without extra dependencies."""
    total = len(true_labels)
    correct = sum(
        1 for true, predicted in zip(true_labels, predicted_labels)
        if true == predicted
    )
    accuracy = correct / total if total else 0.0

    per_class = {}
    weighted_precision = 0.0
    weighted_recall = 0.0
    weighted_f1 = 0.0
    true_counts = Counter(true_labels)

    for label in LABEL_ORDER:
        true_positive = sum(
            1 for true, predicted in zip(true_labels, predicted_labels)
            if true == label and predicted == label
        )
        false_positive = sum(
            1 for true, predicted in zip(true_labels, predicted_labels)
            if true != label and predicted == label
        )
        false_negative = sum(
            1 for true, predicted in zip(true_labels, predicted_labels)
            if true == label and predicted != label
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
        support = true_counts[label]
        weight = support / total if total else 0.0

        weighted_precision += precision * weight
        weighted_recall += recall * weight
        weighted_f1 += f1 * weight

        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

    return {
        "accuracy": accuracy,
        "precision": weighted_precision,
        "recall": weighted_recall,
        "f1": weighted_f1,
        "per_class": per_class,
    }


def print_confusion_matrix(matrix):
    print("\nConfusion Matrix")
    print("Actual \\ Predicted | Negative | Neutral | Positive")
    print("-" * 52)
    for true_label in LABEL_ORDER:
        row = matrix[true_label]
        print(
            f"{true_label:<18} | "
            f"{row['Negative']:^8} | "
            f"{row['Neutral']:^7} | "
            f"{row['Positive']:^8}"
        )


def confusion_matrix_dataframe(matrix):
    dataframe = pd.DataFrame(matrix).T
    dataframe = dataframe.reindex(index=LABEL_ORDER, columns=LABEL_ORDER, fill_value=0)
    dataframe.index.name = "Actual / Predicted"
    return dataframe


def metrics_table_markdown(per_class_metrics):
    lines = [
        "| Class | Precision | Recall | F1 Score | Support |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for label in LABEL_ORDER:
        metrics = per_class_metrics[label]
        lines.append(
            f"| {label} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | "
            f"{metrics['f1']:.4f} | {metrics['support']} |"
        )
    return "\n".join(lines)


def save_reports(
    metrics,
    confusion_matrix,
    runtime_seconds,
    sample_size,
    requested_sample_size,
    random_state,
    balanced,
    dataset_size,
):
    REPORTS_DIR.mkdir(exist_ok=True)
    confusion_df = confusion_matrix_dataframe(confusion_matrix)
    confusion_df.to_csv(CONFUSION_MATRIX_PATH)

    output = {
        "model": MODEL_ID,
        "dataset": str(DATASET_PATH),
        "dataset_rows_after_cleaning": dataset_size,
        "requested_sample_size": requested_sample_size,
        "evaluation_sample_size": sample_size,
        "random_state": random_state,
        "balanced_sampling": balanced,
        "dataset_label_mapping": {
            "-1": "Negative",
            "0": "Neutral",
            "1": "Positive",
        },
        "model_output_mapping": {
            "0": "Negative",
            "1": "Neutral",
            "2": "Positive",
        },
        "metrics": {
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
        },
        "per_class_metrics": metrics["per_class"],
        "confusion_matrix": confusion_matrix,
        "evaluation_runtime_seconds": runtime_seconds,
    }

    with METRICS_PATH.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=4)

    summary = f"""# Production Model Evaluation Summary

## Model Evaluated

`{MODEL_ID}`

## Dataset Used

`{DATASET_PATH}`

Rows after cleaning: **{dataset_size}**

Requested sample size: **{requested_sample_size}**

Actual evaluation sample size: **{sample_size}**

Balanced sampling used: **{"Yes" if balanced else "No"}**

Random state: **{random_state}**

## Label Mappings

Dataset label mapping:

```text
-1 = Negative
 0 = Neutral
 1 = Positive
```

Production model output mapping:

```text
0 = Negative
1 = Neutral
2 = Positive
```

## Overall Metrics

| Metric | Score |
| --- | ---: |
| Accuracy | {metrics['accuracy']:.4f} |
| Precision | {metrics['precision']:.4f} |
| Recall | {metrics['recall']:.4f} |
| F1 Score | {metrics['f1']:.4f} |

## Per-Class Metrics

{metrics_table_markdown(metrics['per_class'])}

## Confusion Matrix

| Actual / Predicted | Negative | Neutral | Positive |
| --- | ---: | ---: | ---: |
| Negative | {confusion_matrix['Negative']['Negative']} | {confusion_matrix['Negative']['Neutral']} | {confusion_matrix['Negative']['Positive']} |
| Neutral | {confusion_matrix['Neutral']['Negative']} | {confusion_matrix['Neutral']['Neutral']} | {confusion_matrix['Neutral']['Positive']} |
| Positive | {confusion_matrix['Positive']['Negative']} | {confusion_matrix['Positive']['Neutral']} | {confusion_matrix['Positive']['Positive']} |

## Short Interpretation

This evaluation measures the current production transformer model on the cleaned Twitter sentiment dataset. Accuracy shows the total share of correct predictions. Precision shows how reliable each predicted sentiment class is. Recall shows how many true examples of each class the model was able to find. F1 Score balances precision and recall, which is useful when sentiment classes are uneven or when false positives and false negatives both matter.

If balanced sampling is enabled, each sentiment class contributes equally to the evaluation sample. This makes the result easier to compare across classes. If balanced sampling is disabled, the evaluation reflects the natural distribution of the dataset.

The model is a general social-media sentiment model, so performance can be affected by noisy tweets, sarcasm, political language, short text, slang, and differences between this dataset's label definitions and the model's original training data.

Evaluation runtime: **{runtime_seconds:.2f} seconds**
"""

    with SUMMARY_PATH.open("w", encoding="utf-8") as file:
        file.write(summary)


def evaluate(evaluation_data):
    true_labels = []
    predicted_labels = []
    total = len(evaluation_data)

    for index, row in evaluation_data.iterrows():
        prediction = predict_sentiment(row[TEXT_COLUMN])
        true_labels.append(row["true_label"])
        predicted_labels.append(prediction["label"])

        if (index + 1) % 25 == 0 or (index + 1) == total:
            print(f"Evaluated {index + 1}/{total} rows")

    return true_labels, predicted_labels


def main():
    args = parse_args()
    requested_sample_size = normalize_sample_size(args.sample_size)

    start_time = time.perf_counter()
    dataframe = load_and_clean_dataset()
    evaluation_data = sample_dataset(
        dataframe=dataframe,
        sample_size=requested_sample_size,
        random_state=args.random_state,
        balanced=args.balanced,
    )

    print(f"\nModel: {MODEL_ID}")
    print(f"Dataset rows after cleaning: {len(dataframe)}")
    print(f"Requested sample size: {requested_sample_size}")
    print(f"Evaluation sample size: {len(evaluation_data)}")
    print(f"Balanced sampling: {'Yes' if args.balanced else 'No'}")
    print("Running production model evaluation...")

    true_labels, predicted_labels = evaluate(evaluation_data)

    runtime_seconds = time.perf_counter() - start_time
    metrics = compute_metrics(true_labels, predicted_labels)
    confusion_matrix = build_confusion_matrix(true_labels, predicted_labels)

    print("\nFinal Evaluation Results")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    print(f"Runtime:   {runtime_seconds:.2f} seconds")

    print("\nPer-Class Metrics")
    for label in LABEL_ORDER:
        class_metrics = metrics["per_class"][label]
        print(
            f"{label:<8} "
            f"Precision={class_metrics['precision']:.4f} "
            f"Recall={class_metrics['recall']:.4f} "
            f"F1={class_metrics['f1']:.4f} "
            f"Support={class_metrics['support']}"
        )

    print_confusion_matrix(confusion_matrix)

    save_reports(
        metrics=metrics,
        confusion_matrix=confusion_matrix,
        runtime_seconds=runtime_seconds,
        sample_size=len(evaluation_data),
        requested_sample_size=requested_sample_size,
        random_state=args.random_state,
        balanced=args.balanced,
        dataset_size=len(dataframe),
    )

    print(f"\nSaved metrics to: {METRICS_PATH}")
    print(f"Saved summary to: {SUMMARY_PATH}")
    print(f"Saved confusion matrix to: {CONFUSION_MATRIX_PATH}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print(f"Evaluation stopped: {error}")
