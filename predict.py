import os
import pickle
import time
from pathlib import Path

from preprocessing import preprocess_text

MODEL_PATH = Path("models/traditional_model.pkl")
VECTORIZER_PATH = Path("models/tfidf_vectorizer.pkl")
MODEL_NAME = "TF-IDF customer sentiment fallback"


def model_files_exist():
    return MODEL_PATH.exists() and VECTORIZER_PATH.exists()


def load_traditional_assets():
    if not model_files_exist():
        raise FileNotFoundError("Traditional model files were not found in the models folder.")
    with MODEL_PATH.open("rb") as model_file:
        model = pickle.load(model_file)
    with VECTORIZER_PATH.open("rb") as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    return model, vectorizer


def predict_sentiment(text):
    """Predict sentiment with the optional local TF-IDF model."""
    start_time = time.perf_counter()
    steps = preprocess_text(text)
    model, vectorizer = load_traditional_assets()
    features = vectorizer.transform([steps["final_text"]])
    sentiment = model.predict(features)[0]

    confidence = 0.0
    if hasattr(model, "predict_proba"):
        confidence = float(model.predict_proba(features).max() * 100)

    return {
        "sentiment": str(sentiment),
        "confidence": confidence,
        "processing_time": time.perf_counter() - start_time,
        "model_name": MODEL_NAME,
        "processed_text": steps["final_text"],
    }
