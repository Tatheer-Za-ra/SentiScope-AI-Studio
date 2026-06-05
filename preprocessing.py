import re
import string

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "while", "with", "to", "from", "in", "on", "at",
    "for", "of", "by", "is", "am", "are", "was", "were", "be", "been", "being", "it", "this",
    "that", "these", "those", "i", "you", "he", "she", "we", "they", "me", "my", "your", "our",
    "their", "as", "so", "very", "really", "just", "about", "into", "than", "then", "too"
}

POSITIVE_WORDS = {
    "good", "great", "excellent", "amazing", "awesome", "love", "liked", "like", "best", "happy",
    "satisfied", "fast", "quick", "helpful", "smooth", "beautiful", "easy", "reliable", "perfect",
    "wonderful", "positive", "improved", "friendly", "recommend", "polished", "useful", "delightful"
}

NEGATIVE_WORDS = {
    "bad", "poor", "terrible", "awful", "hate", "hated", "worst", "sad", "angry", "slow",
    "bug", "bugs", "crash", "crashing", "broken", "confusing", "difficult", "late", "delay",
    "delayed", "negative", "unhappy", "disappointed", "problem", "problems", "issue", "issues",
    "expensive", "missing", "failed", "failure"
}


def preprocess_text(text):
    """Return beginner-friendly preprocessing steps for display in the app."""
    original_text = "" if text is None else str(text)
    lowercased_text = original_text.lower()
    cleaned_text = lowercased_text.translate(str.maketrans("", "", string.punctuation))
    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()
    raw_tokens = cleaned_text.split()
    tokens = [word for word in raw_tokens if word not in STOPWORDS]
    final_text = " ".join(tokens)

    return {
        "original_text": original_text,
        "lowercased_text": lowercased_text,
        "cleaned_text": cleaned_text,
        "raw_tokens": raw_tokens,
        "tokens": tokens,
        "stopwords_removed": tokens,
        "final_text": final_text,
    }
