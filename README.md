# SentiScope AI Studio

**SentiScope AI Studio** is a client-ready AI-powered customer feedback and review sentiment analytics studio. It transforms unstructured feedback into actionable sentiment insights using a fine-tuned transformer model and an interactive SaaS dashboard.

**Status:** Development complete locally. Deployment pending.

## Overview

SentiScope AI Studio helps product teams, customer support teams, social media managers, freelancers, and small businesses understand customer sentiment at scale.

Users can analyze individual feedback messages, upload CSV files for batch sentiment analysis, view sentiment insights through an interactive dashboard, and export clean prediction results for reporting.

## Key Features

**Single Text Analysis**

- Analyze one review, comment, tweet, or feedback message
- Show final sentiment, confidence score, and probability breakdown
- Provide a natural-language insight for the prediction
- Highlight positive and negative sentiment keywords

**Batch CSV Analysis**

- Upload CSV files containing customer feedback
- Auto-detect likely text columns such as `text`, `review`, `comment`, `feedback`, `tweet`, `content`, and `clean_text`
- Select the text column manually when needed
- Preview selected feedback values before analysis
- Export clean prediction results as CSV

**Insights Dashboard**

- Total feedback analyzed
- Positive, Neutral, and Negative feedback counts
- Average confidence and dominant sentiment
- Sentiment distribution charts
- Confidence distribution chart
- Key sentiment insights

**Product UI**

- Streamlit-based SaaS-style interface
- Light/Dark theme support
- Responsive dashboard layout
- Client-friendly wording

## Demo Status

The app currently runs locally with Streamlit.

**Live deployment:** Coming soon.

No public deployment link is available yet.

## Tech Stack

- Python
- Streamlit
- Pandas
- Plotly
- Hugging Face Transformers
- PyTorch
- Twitter-RoBERTa sentiment transformer

## Project Pages

### Home / Overview
Introduces the product and workflow.

### Analyze Text
Test single-text sentiment analysis with sample reviews or paste your own feedback. See final sentiment, confidence score, probability breakdown, and keyword-level explanation.

### Batch Analysis
Upload a CSV file containing customer feedback. Select the text column, generate predictions, preview results, and download clean output.

### Insights Dashboard
Visualize sentiment trends with KPI cards (total feedback, positive/neutral/negative counts), sentiment distribution charts, confidence histograms, and detailed results.

### About
Product overview, technical details about the AI sentiment engine, and recommended workflow.

## AI Model

**Model:** Fine-tuned Twitter-RoBERTa sentiment transformer  
**Optimized for:** Customer reviews, support comments, survey responses, social media text  
**Sentiment classes:** Positive, Neutral, Negative  
**Local model path:** `models/sentiment-transformer/`

If the local fine-tuned model is not available, the app falls back to:

```text
cardiffnlp/twitter-roberta-base-sentiment-latest
```

## Model Performance

**Balanced Quality Score: 85.58%**

Balanced Quality Score is based on macro F1 and measures how consistently the model performs across Positive, Neutral, and Negative sentiment classes.

| Metric | Score |
| --- | ---: |
| Accuracy | 85.53% |
| Macro Precision | 85.88% |
| Macro Recall | 85.53% |
| Balanced Quality Score / Macro F1 | 85.58% |

## CSV Format

For batch analysis, upload a CSV file with a column containing customer feedback text. Common supported column names:

```
text, review, comment, feedback, tweet, content, message, clean_text
```

Example:

| review |
| --- |
| The product quality is excellent and delivery was fast. |
| The app crashes whenever I try to checkout. |

The app will auto-detect the text column. If multiple text-like columns exist, you can choose which one to analyze.

## Local Setup

Clone the repository:

```bash
git clone https://github.com/Tatheer-Za-ra/SentiScope-AI-Studio.git
cd SentiScope-AI-Studio
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
streamlit run app.py
```

Or on systems where streamlit is installed as a package:
```bash
python -m streamlit run app.py
```

The app will open at `http://localhost:8501`

## Fine-Tuned Model Note

The fine-tuned model folder is ignored from GitHub because transformer artifacts can be large.

To use the fine-tuned model locally, place the model files in:

```
models/sentiment-transformer/
```

If that folder is missing or incomplete, the app automatically falls back to the base Hugging Face model:

```
cardiffnlp/twitter-roberta-base-sentiment-latest
```

Do not commit local model files to GitHub.

## Training and Evaluation

`train_transformer.py` is used for fine-tuning. `evaluate_model.py` is used for evaluation.

Example training command:

```bash
python train_transformer.py --epochs 1 --batch-size 4 --max-rows 30000
```

Example evaluation command:

```bash
python evaluate_model.py --sample-size 5000 --balanced
```

Training can be slow on CPU and is optional for running the app if the fallback model is acceptable.

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

## Deployment Status

Deployment is not completed yet.

Planned deployment options:

- Streamlit Community Cloud
- Hugging Face Spaces
- Render
- Docker-based deployment

No public live demo link is available at this time.

## Repository Hygiene

Large local files such as datasets, model artifacts, generated reports, exports, logs, virtual environments, caches, and secrets are excluded using `.gitignore`.

## Future Improvements

- Public deployment
- Authentication for saved dashboards
- More domain-specific feedback datasets
- API endpoint for product integrations
- Better explainability
- PDF or Excel export

## License

License not specified yet.
