import os
import time
from functools import lru_cache
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
os.environ.setdefault("USE_TORCHVISION", "0")

MODEL_ID = "cardiffnlp/twitter-roberta-base-sentiment-latest"
LOCAL_MODEL_PATH = Path("models/sentiment-transformer")
LOCAL_MODEL_DISPLAY_NAME = "Fine-tuned Transformer Sentiment Model"
BASE_MODEL_DISPLAY_NAME = "Base Twitter-RoBERTa Sentiment Model"

LABEL_MAP = {
    0: "Negative",
    1: "Neutral",
    2: "Positive",
}

PROBABILITY_KEYS = {
    0: "negative",
    1: "neutral",
    2: "positive",
}


def local_model_available():
    """A fine-tuned checkpoint is usable only after training saves config.json."""
    return LOCAL_MODEL_PATH.exists() and (LOCAL_MODEL_PATH / "config.json").exists()


@lru_cache(maxsize=1)
def load_model():
    """Load the fine-tuned local model if available, otherwise use the base model."""
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except Exception as error:
        raise RuntimeError(
            "Transformer dependencies are missing. Install requirements.txt first."
        ) from error

    model_source = str(LOCAL_MODEL_PATH) if local_model_available() else MODEL_ID

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_source)
        model = AutoModelForSequenceClassification.from_pretrained(model_source)
        model.eval()
    except Exception as error:
        raise RuntimeError(
            "Could not load the sentiment model. If the local fine-tuned model is "
            "missing or incomplete, remove models/sentiment-transformer and retry "
            "so the app can fall back to the base Hugging Face model."
        ) from error

    display_name = LOCAL_MODEL_DISPLAY_NAME if local_model_available() else BASE_MODEL_DISPLAY_NAME
    return tokenizer, model, model_source, display_name


def predict_sentiment(text):
    """Predict Positive, Negative, or Neutral sentiment for one text input."""
    clean_text = "" if text is None else str(text).strip()
    if not clean_text:
        raise ValueError("Please enter customer feedback text before analyzing sentiment.")

    start_time = time.perf_counter()

    try:
        import torch
    except Exception as error:
        raise RuntimeError(
            "PyTorch is required for transformer inference. Install requirements.txt first."
        ) from error

    tokenizer, model, model_source, display_name = load_model()

    encoded_text = tokenizer(
        clean_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256,
    )

    with torch.no_grad():
        output = model(**encoded_text)
        scores = torch.softmax(output.logits, dim=1)[0]

    predicted_index = int(torch.argmax(scores).item())
    probabilities = {
        PROBABILITY_KEYS[index]: float(scores[index].item())
        for index in range(len(LABEL_MAP))
    }

    return {
        "label": LABEL_MAP[predicted_index],
        "confidence": probabilities[PROBABILITY_KEYS[predicted_index]],
        "probabilities": probabilities,
        "inference_time": time.perf_counter() - start_time,
        "model_name": model_source,
        "display_name": display_name,
    }
