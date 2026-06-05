import os
import time
from functools import lru_cache

os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
os.environ.setdefault("USE_TORCHVISION", "0")

MODEL_ID = "cardiffnlp/twitter-roberta-base-sentiment-latest"

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

@lru_cache(maxsize=1)
def load_model():
    """Load the production transformer once and reuse it for later predictions."""
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except Exception as error:
        raise RuntimeError(
            "Transformer dependencies are missing. Install requirements.txt first."
        ) from error

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
        model.eval()
    except Exception as error:
        raise RuntimeError(
            "Could not load the production sentiment model. Check your internet "
            "connection for the first download, or verify the Hugging Face cache."
        ) from error

    return tokenizer, model


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

    tokenizer, model = load_model()

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
        "model_name": MODEL_ID,
    }
