import os
import time

os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
os.environ.setdefault("USE_TORCHVISION", "0")

HF_MODEL_ID = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"
MODEL_NAME = "DistilBERT customer sentiment engine"

_tokenizer = None
_model = None


def _load_model():
    """Load tokenizer and transformer model lazily so the app starts faster."""
    global _tokenizer, _model
    if _tokenizer is not None and _model is not None:
        return _tokenizer, _model

    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except Exception as error:
        raise RuntimeError(f"Required transformer packages are not installed: {error}") from error

    _tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)
    _model = AutoModelForSequenceClassification.from_pretrained(HF_MODEL_ID)
    _model.eval()
    return _tokenizer, _model


def _normalize_label(label):
    value = str(label).lower()
    if "positive" in value:
        return "Positive"
    if "negative" in value:
        return "Negative"
    return "Neutral"


def predict_distilbert_sentiment(text):
    """Predict sentiment using a text-only Hugging Face DistilBERT model."""
    start_time = time.perf_counter()
    clean_text = "" if text is None else str(text).strip()
    if not clean_text:
        return {
            "sentiment": "Neutral",
            "confidence": 0.0,
            "processing_time": 0.0,
            "model_name": MODEL_NAME,
            "error": "Please enter text before running sentiment analysis.",
        }

    try:
        import torch
        tokenizer, model = _load_model()
        encoded = tokenizer(clean_text, return_tensors="pt", truncation=True, padding=True, max_length=256)
        with torch.no_grad():
            outputs = model(**encoded)
            probabilities = torch.softmax(outputs.logits, dim=1)[0]
            predicted_index = int(torch.argmax(probabilities).item())
            confidence = float(probabilities[predicted_index].item() * 100)
        label = model.config.id2label.get(predicted_index, str(predicted_index))
        return {
            "sentiment": _normalize_label(label),
            "confidence": confidence,
            "processing_time": time.perf_counter() - start_time,
            "model_name": MODEL_NAME,
            "error": None,
        }
    except Exception as error:
        return {
            "sentiment": "Unavailable",
            "confidence": 0.0,
            "processing_time": time.perf_counter() - start_time,
            "model_name": MODEL_NAME,
            "error": f"DistilBERT sentiment engine is unavailable: {error}",
        }
