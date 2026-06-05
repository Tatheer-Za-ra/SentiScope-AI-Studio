# SentiScope AI Studio

SentiScope AI Studio is a client-ready sentiment analytics dashboard for customer feedback, product reviews, support comments, survey responses, and social media text.

The product converts unstructured feedback into clear sentiment insights using an AI-powered Streamlit dashboard. It is designed for portfolio presentation, client demos, and practical review analytics workflows.

## Product Positioning

**SentiScope AI Studio** helps teams understand customer mood at scale. Users can analyze a single comment, upload a CSV file of customer feedback, view interactive analytics, and export prediction results for business reporting.

## Core Features

- Single text sentiment analysis
- Batch CSV sentiment analysis
- Automatic text-column detection
- Customer feedback analytics dashboard
- Sentiment distribution charts
- Confidence distribution chart
- CSV export of prediction results
- Keyword-level sentiment explanation
- Preprocessing pipeline preview
- Modern dark SaaS-style interface

## Pages

### Overview
Introduces the product and the main workflow for feedback analytics.

### Single Analysis
Allows users to analyze one review, tweet, support message, or comment. The page shows sentiment, confidence, processing time, keyword-level explanation, and preprocessing steps.

### Batch CSV Analysis
Allows users to upload a CSV file, select the text column, generate predictions, preview results, and download the analyzed CSV.

### Analytics Dashboard
Displays KPI cards, sentiment distribution charts, confidence distribution, and recent prediction results from the latest batch analysis.

### About
Explains the product use case and recommended workflow for client-facing demos.

## Dataset Format

For batch analysis, upload a CSV file containing a column with customer feedback text. Common supported column names include:

```text
text
tweet
review
comment
content
message
clean_text
feedback
```

If several text-like columns exist, the app lets the user choose the correct column manually.

## Local Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
streamlit run app.py
```

## Production Sentiment Model

SentiScope AI Studio uses one production transformer sentiment model:

```text
cardiffnlp/twitter-roberta-base-sentiment-latest
```

The model is loaded through Hugging Face Transformers and returns:

- sentiment label
- confidence score
- probability distribution
- inference time

The first run may take longer because the model needs to download from Hugging Face. Later runs use the local Hugging Face cache.

## Project Structure

```text
sentiscope-ai-studio/
├── app.py
├── model_service.py
├── preprocessing.py
├── requirements.txt
├── README.md
├── .gitignore
├── .streamlit/
│   └── config.toml
├── data/
│   └── .gitkeep
└── exports/
    └── .gitkeep
```

## GitHub Notes

The repository should include source code, configuration, documentation, and empty folder placeholders. It should not include datasets, exported CSV results, Hugging Face caches, virtual environments, or local logs.

## Portfolio Demo Flow

1. Open the Overview page.
2. Test a sample review in Single Analysis.
3. Upload a small CSV file in Batch CSV Analysis.
4. Generate predictions and download the result CSV.
5. Open Analytics Dashboard to view sentiment insights.

## Deployment Notes

This project is ready for later deployment on Streamlit Community Cloud or a similar hosting platform. For deployment, ensure dependencies install correctly, the model can download from Hugging Face, and local datasets or generated exports are not committed to the repository.
