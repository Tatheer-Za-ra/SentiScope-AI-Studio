# SentiScope AI Studio

**SentiScope AI Studio** is a client-ready AI-powered customer feedback and review sentiment analytics studio. It transforms unstructured feedback into actionable sentiment insights using a fine-tuned transformer model and an interactive SaaS dashboard.

Perfect for product teams, customer success teams, and businesses that want to understand customer sentiment at scale.

## Core Features

- **Single Text Analysis** – Analyze a single review, comment, or support message instantly
- **Batch CSV Analysis** – Upload CSV files with customer feedback and generate sentiment predictions  
- **Auto-Column Detection** – Automatically detects text columns (review, comment, feedback, tweet, etc.)
- **Interactive Dashboard** – Visualize sentiment distribution, confidence scores, and key insights
- **CSV Export** – Download clean results for business reporting and further analysis
- **Keyword-Level Explanation** – Identifies positive and negative keywords in customer feedback
- **Fine-tuned AI Model** – Uses a Twitter-RoBERTa sentiment transformer optimized for customer text
- **Light/Dark Theme** – Modern SaaS interface with customizable appearance

## Pages & Navigation

### Overview
Introduces the product and workflow. Quick links to test analysis, upload CSV, or review the dashboard.

### Analyze Text
Test single-text sentiment analysis with sample reviews or paste your own feedback. See final sentiment, confidence score, probability breakdown, and keyword-level explanation.

### Batch Analysis
Upload a CSV file containing customer feedback. Select the text column, generate predictions, preview results, and download clean output.

### Insights Dashboard
Visualize sentiment trends with KPI cards (total feedback, positive/neutral/negative counts), sentiment distribution charts, confidence histograms, and detailed results.

### About
Product overview, technical details about the AI sentiment engine, and recommended workflow.

## AI Sentiment Engine

**Model:** Fine-tuned Twitter-RoBERTa sentiment transformer  
**Optimized for:** Customer reviews, support comments, survey responses, social media text  
**Output:** Sentiment label (Positive/Neutral/Negative), confidence score, probability distribution

### Performance

**Balanced Quality Score: 85.58%**

The Balanced Quality Score measures model performance equally across all three sentiment classes. It is based on macro F1, a standard machine learning metric that treats Positive, Neutral, and Negative sentiments fairly. This score was measured on a held-out test set.

Detailed metrics:
- Accuracy: 85.53%
- Macro Precision: 85.88%
- Macro Recall: 85.53%
- Macro F1: 85.58%

## CSV Format

For batch analysis, upload a CSV file with a column containing customer feedback text. Common supported column names:

```
text, review, comment, feedback, tweet, content, message, clean_text
```

Example CSV structure:
```
review_text,customer_name,date
"The product is amazing!","John","2024-01-15"
"Disappointed with quality","Jane","2024-01-16"
```

The app will auto-detect the text column. If multiple text-like columns exist, you can choose which one to analyze.

## Local Setup

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run the App
```bash
streamlit run app.py
```

Or on systems where streamlit is installed as a package:
```bash
python -m streamlit run app.py
```

The app will open at `http://localhost:8501`

## Model Files

### Fine-tuned Model
The app first attempts to load a fine-tuned local model from:
```
models/sentiment-transformer/
```

If this folder is missing or incomplete, the app automatically falls back to the base Hugging Face model:
```
cardiffnlp/twitter-roberta-base-sentiment-latest
```

### To Train or Update the Model

1. Place your training data at:
   ```
   data/Twitter_Data.csv
   ```

2. Run:
   ```bash
   python train_transformer.py --epochs 2 --batch-size 8
   ```

3. The trained model saves to:
   ```
   models/sentiment-transformer/
   ```

### To Evaluate the Model

```bash
python evaluate_model.py --sample-size 5000 --balanced
```

## Project Structure

```
sentiscope-ai-studio/
├── app.py                          # Main Streamlit app
├── model_service.py                # Model loading & prediction
├── preprocessing.py                # Text preprocessing utilities
├── train_transformer.py            # Fine-tuning script
├── evaluate_model.py               # Model evaluation script
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── .streamlit/
│   └── config.toml                 # Streamlit config
├── data/                           # Training datasets (not committed)
├── exports/                        # Generated CSV results
├── models/
│   └── sentiment-transformer/      # Fine-tuned model (not committed)
└── reports/                        # Analysis reports
```

## Deployment

This project is deployment-ready for:
- **Local demos** – Run locally with `streamlit run app.py`
- **Streamlit Cloud** – Deploy via Streamlit Community Cloud
- **Docker** – Containerize for cloud deployment
- **VPS/Linux Server** – Run with Gunicorn or similar

### Deployment Notes
- Dependencies install from `requirements.txt`
- Model loads from `models/sentiment-transformer/` or falls back to Hugging Face
- Large files (datasets, exports, model checkpoints) are not committed to the repository
- The app requires internet access on first run to download the base model (cached locally after)

## Performance & Inference

- **Single text inference:** ~0.5–2 seconds (depends on text length and GPU availability)
- **Batch processing:** 50 rows ~20–60 seconds on standard hardware
- **Model size:** ~400MB for transformer model + dependencies

First run may take longer as dependencies and the base model are downloaded. Subsequent runs use cached models.

## Support & Notes

- All user-facing text is client-friendly and product-focused
- Preprocessing steps are handled internally; users see only final results
- Keyword highlighting is for context; final predictions come from the AI model
- CSV export includes text, sentiment, confidence, and probability scores
- Light and dark themes are available for comfortable viewing

## Repository Notes

This repository includes source code, configuration, and documentation. It does NOT include:
- Training datasets
- Generated CSV exports
- Model checkpoint files (use `train_transformer.py` to create)
- Virtual environments
- Large transformer model binaries
- Local log files
