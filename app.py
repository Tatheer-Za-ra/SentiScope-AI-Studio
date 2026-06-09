import html
import re
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components

from model_service import MODEL_ID, predict_sentiment
from preprocessing import NEGATIVE_WORDS, POSITIVE_WORDS, STOPWORDS

st.set_page_config(
    page_title="SentiScope AI Studio",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

EXPORT_DIR = Path("exports")
EXPORT_DIR.mkdir(exist_ok=True)

SAMPLE_TEXTS = {
    "Positive Review": "The support team responded quickly and the product works beautifully after the latest update.",
    "Negative Review": "The app keeps crashing during checkout and customer support has not replied for two days.",
    "Neutral Review": "I installed the application today and explored the dashboard and settings page.",
    "Mixed Review": "The interface looks polished, but the delivery tracking is slow and confusing.",
}

PRODUCT_POSITIONING = "Transform customer feedback into actionable sentiment insights."
PRODUCT_DESCRIPTION = "Analyze reviews, comments, and customer feedback using an AI-powered sentiment engine."
MODEL_INFO_TITLE = "AI Sentiment Engine"
MODEL_INFO_DETAIL = "Fine-tuned Twitter-RoBERTa sentiment transformer"
MODEL_EVALUATION_TEXT = "Balanced Quality Score: 85.58%"
MODEL_EVALUATION_HELP = (
    "Balanced Quality Score summarizes performance across Positive, Neutral, "
    "and Negative sentiment classes. It is based on macro F1 and was measured "
    "on a held-out test set."
)


def get_theme_palette(theme_name):
    if theme_name == "Light Mode":
        return {
            "bg": "#F8FAFC",
            "sidebar": "#FFFFFF",
            "panel": "#FFFFFF",
            "panel_soft": "#F1F5F9",
            "border": "#E2E8F0",
            "text": "#0F172A",
            "muted": "#475569",
            "accent": "#2563EB",
            "accent_2": "#1D4ED8",
            "green": "#16A34A",
            "red": "#DC2626",
            "blue": "#64748B",
            "shadow": "0 18px 36px rgba(15,23,42,.10)",
            "input_bg": "#FFFFFF",
            "grid": "#E2E8F0",
        }
    return {
        "bg": "#0B1120",
        "sidebar": "#0B1120",
        "panel": "#111827",
        "panel_soft": "#172033",
        "border": "#1E293B",
        "text": "#F8FAFC",
        "muted": "#CBD5E1",
        "accent": "#38BDF8",
        "accent_2": "#2563EB",
        "green": "#22C55E",
        "red": "#EF4444",
        "blue": "#60A5FA",
        "shadow": "0 18px 40px rgba(0,0,0,.22)",
        "input_bg": "#0F172A",
        "grid": "#263A59",
    }


def load_css(theme_name):
    palette = get_theme_palette(theme_name)
    css = f"""
        <style>
        :root {{
            --bg: {palette["bg"]};
            --sidebar: {palette["sidebar"]};
            --panel: {palette["panel"]};
            --panel-soft: {palette["panel_soft"]};
            --border: {palette["border"]};
            --text: {palette["text"]};
            --muted: {palette["muted"]};
            --accent: {palette["accent"]};
            --accent-2: {palette["accent_2"]};
            --green: {palette["green"]};
            --red: {palette["red"]};
            --blue: {palette["blue"]};
            --shadow: {palette["shadow"]};
            --input-bg: {palette["input_bg"]};
        }}
        .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {{
            background: var(--bg) !important;
            color: var(--text) !important;
        }}
        [data-testid="stSidebar"] {{
            background: var(--sidebar) !important;
            border-right: 1px solid var(--border);
        }}
        h1, h2, h3, h4, h5, h6, p, li, label, span, div {{
            color: var(--text);
        }}
        .hero {{
            padding: 2rem;
            border: 1px solid var(--border);
            border-radius: 18px;
            background: linear-gradient(135deg, var(--panel) 0%, var(--panel-soft) 100%);
            margin-bottom: 1.5rem;
        }}
        .hero-title {{
            font-size: 3rem;
            font-weight: 800;
            letter-spacing: 0;
            margin-bottom: .4rem;
        }}
        .subtitle {{
            color: var(--muted);
            font-size: 1.08rem;
            line-height: 1.7;
        }}
        .section-title {{
            font-size: 1.45rem;
            font-weight: 750;
            margin: 1.2rem 0 .7rem 0;
        }}
        .section-spacer {{
            height: 1.5rem;
        }}
        .card {{
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 1.1rem;
            box-shadow: var(--shadow);
            min-height: 100px;
        }}
        .metric-card {{
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 1rem;
        }}
        .small-label {{ color: var(--muted); font-size: .85rem; }}
        .big-value {{ font-size: 1.65rem; font-weight: 800; margin-top: .25rem; }}
        .insight-text {{ color: var(--muted); line-height: 1.65; }}
        .positive {{ border-left: 5px solid var(--green); }}
        .negative {{ border-left: 5px solid var(--red); }}
        .neutral {{ border-left: 5px solid var(--blue); }}
        .highlight-pos {{ background: rgba(22,163,74,.16); color: var(--green); padding: 2px 6px; border-radius: 7px; font-weight: 700; }}
        .highlight-neg {{ background: rgba(220,38,38,.16); color: var(--red); padding: 2px 6px; border-radius: 7px; font-weight: 700; }}
        .highlight-neutral {{ color: var(--muted); }}
        .prob-row {{ margin: .65rem 0; }}
        .prob-label {{ display:flex; justify-content:space-between; color:var(--text); font-size:.92rem; margin-bottom:.3rem; }}
        .prob-track {{ height: 9px; border-radius: 999px; background: var(--panel-soft); border: 1px solid var(--border); overflow: hidden; }}
        .prob-fill {{ height: 100%; border-radius: 999px; }}
        .badge {{ display:inline-block; padding:.2rem .55rem; border-radius:999px; font-weight:700; font-size:.82rem; }}
        .badge-positive {{ color: var(--green); background: rgba(22,163,74,.12); border: 1px solid rgba(22,163,74,.30); }}
        .badge-neutral {{ color: var(--blue); background: rgba(100,116,139,.14); border: 1px solid rgba(100,116,139,.32); }}
        .badge-negative {{ color: var(--red); background: rgba(220,38,38,.12); border: 1px solid rgba(220,38,38,.30); }}
        .stButton > button, .stDownloadButton > button {{
            background: linear-gradient(135deg, var(--accent), var(--accent-2)) !important;
            color: white !important;
            border: 1px solid rgba(255,255,255,.18) !important;
            border-radius: 12px !important;
            font-weight: 700 !important;
            padding: .65rem 1rem !important;
        }}
        .stButton > button:hover, .stDownloadButton > button:hover {{
            border-color: var(--accent) !important;
            filter: brightness(1.08);
        }}
        textarea, input, [data-baseweb="select"] > div {{
            background: var(--input-bg) !important;
            color: var(--text) !important;
            border-color: var(--border) !important;
        }}
        textarea::placeholder, input::placeholder {{ color: var(--muted) !important; opacity: 1 !important; }}
        textarea {{ caret-color: var(--accent) !important; }}
        [data-testid="stDataFrame"] {{ border: 1px solid var(--border); border-radius: 12px; overflow: hidden; }}
        [data-testid="stMetric"] {{
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: .85rem;
            box-shadow: var(--shadow);
        }}
        [data-testid="stAlert"] {{
            border-radius: 12px;
            border: 1px solid var(--border);
        }}
        
                /* File uploader main wrapper */
        [data-testid="stFileUploader"] {{
            background: transparent !important;
            color: var(--text) !important;
        }}

        [data-testid="stFileUploader"] label,
        [data-testid="stFileUploader"] label p,
        [data-testid="stFileUploader"] p {{
            color: var(--text) !important;
        }}

        /* Dropzone before file is selected */
        [data-testid="stFileUploaderDropzone"],
        [data-testid="stFileUploadDropzone"] {{
            background: var(--panel) !important;
            border: 2px dashed var(--border) !important;
            border-radius: 14px !important;
            color: var(--text) !important;
        }}

        [data-testid="stFileUploaderDropzone"] p,
        [data-testid="stFileUploaderDropzone"] span,
        [data-testid="stFileUploaderDropzone"] small,
        [data-testid="stFileUploadDropzone"] p,
        [data-testid="stFileUploadDropzone"] span,
        [data-testid="stFileUploadDropzone"] small {{
            color: var(--text) !important;
        }}

        [data-testid="stFileUploaderDropzone"] small,
        [data-testid="stFileUploadDropzone"] small {{
            color: var(--muted) !important;
        }}

        [data-testid="stFileUploaderDropzone"] svg,
        [data-testid="stFileUploadDropzone"] svg {{
            color: var(--accent) !important;
            fill: var(--accent) !important;
        }}

        /* Browse / upload button */
        [data-testid="stFileUploaderDropzone"] button,
        [data-testid="stFileUploadDropzone"] button {{
            background: var(--panel-soft) !important;
            color: var(--accent) !important;
            border: 1px solid var(--accent) !important;
            border-radius: 10px !important;
            font-weight: 700 !important;
        }}

        /* Uploaded file chip/card after CSV is selected */
        [data-testid="stFileUploaderFile"] {{
            background: var(--panel-soft) !important;
            border: 1px solid var(--border) !important;
            border-radius: 12px !important;
            color: var(--text) !important;
            box-shadow: none !important;
        }}

        /* Uploaded filename and file size */
        [data-testid="stFileUploaderFile"] div,
        [data-testid="stFileUploaderFile"] span,
        [data-testid="stFileUploaderFile"] p,
        [data-testid="stFileUploaderFile"] small {{
            color: var(--text) !important;
        }}

        [data-testid="stFileUploaderFile"] small {{
            color: var(--muted) !important;
        }}

        /* File icon inside selected file chip */
        [data-testid="stFileUploaderFile"] svg {{
            color: var(--accent) !important;
            fill: var(--accent) !important;
        }}

        /* Remove / close button on selected uploaded file */
        [data-testid="stFileUploaderFile"] button,
        [data-testid="stFileUploaderDeleteBtn"] {{
            background: transparent !important;
            color: var(--accent) !important;
            border: 1px solid var(--accent) !important;
            border-radius: 999px !important;
        }}    
        </style>
        """
    st.markdown(css, unsafe_allow_html=True)


def safe_text(value):
    return "" if value is None else str(value)


def sentiment_class(sentiment):
    label = safe_text(sentiment).lower()
    if "positive" in label:
        return "positive"
    if "negative" in label:
        return "negative"
    return "neutral"


def format_confidence(value):
    return f"{float(value) * 100:.2f}%"


def probability_percent(value):
    return f"{float(value) * 100:.1f}%"


def highest_probability_category(probabilities):
    if not probabilities:
        return "Neutral"
    key = max(probabilities, key=probabilities.get)
    return key.title()


def sentiment_badge(sentiment):
    label = safe_text(sentiment).title()
    class_name = sentiment_class(label)
    return f'<span class="badge badge-{class_name}">{html.escape(label)}</span>'


def sentiment_insight(sentiment):
    label = safe_text(sentiment).lower()
    if "positive" in label:
        return "The feedback expresses satisfaction, approval, or a positive customer experience."
    if "negative" in label:
        return "The feedback indicates frustration, dissatisfaction, or a potential customer issue."
    return "The feedback appears mostly informational, balanced, or without strong emotional tone."


# def scroll_to_top():
#     """Safely scroll the page to the top using multiple methods for compatibility."""
#     st.markdown(
#         """
#         <script>
#         try {{
#             window.scrollTo({{ top: 0, behavior: 'instant' }});
#         }} catch (e) {{
#             window.scrollTo(0, 0);
#         }}
#         </script>
#         """,
#         unsafe_allow_html=True,
#     )

def scroll_to_top():
    """Scroll Streamlit page to the top after navigation."""
    components.html(
        """
        <script>
        function forceScrollTop() {
            try {
                const parentWindow = window.parent;
                const parentDoc = parentWindow.document;

                parentWindow.scrollTo(0, 0);

                if (parentDoc.documentElement) {
                    parentDoc.documentElement.scrollTop = 0;
                }

                if (parentDoc.body) {
                    parentDoc.body.scrollTop = 0;
                }

                const possibleScrollContainers = [
                    parentDoc.querySelector('[data-testid="stAppViewContainer"]'),
                    parentDoc.querySelector('[data-testid="stMain"]'),
                    parentDoc.querySelector('section[data-testid="stMain"]'),
                    parentDoc.querySelector('section.main'),
                    parentDoc.querySelector('.main')
                ];

                possibleScrollContainers.forEach(function(container) {
                    if (container) {
                        container.scrollTop = 0;
                    }
                });
            } catch (error) {
                console.log("Scroll-to-top skipped:", error);
            }
        }

        forceScrollTop();
        setTimeout(forceScrollTop, 50);
        setTimeout(forceScrollTop, 150);
        setTimeout(forceScrollTop, 300);
        setTimeout(forceScrollTop, 700);
        </script>
        """,
        height=0,
    )

def render_probability_bars(probabilities):
    rows = [
        ("Negative", probabilities.get("negative", 0.0), "var(--red)"),
        ("Neutral", probabilities.get("neutral", 0.0), "var(--blue)"),
        ("Positive", probabilities.get("positive", 0.0), "var(--green)"),
    ]
    html_rows = []
    for label, value, color in rows:
        width = max(0, min(float(value) * 100, 100))
        html_rows.append(
            f"""
            <div class="prob-row">
                <div class="prob-label"><span>{html.escape(label)}</span><span>{probability_percent(value)}</span></div>
                <div class="prob-track"><div class="prob-fill" style="width:{width:.2f}%; background:{color};"></div></div>
            </div>
            """
        )
    st.markdown("".join(html_rows), unsafe_allow_html=True)


def clean_export_dataframe(results_dataframe):
    export_columns = [
        "original_text",
        "final_sentiment",
        "confidence",
        "negative_probability",
        "neutral_probability",
        "positive_probability",
        "highest_probability_category",
    ]
    return results_dataframe[export_columns].copy()


def render_card(label, value, subtitle=""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="small-label">{html.escape(label)}</div>
            <div class="big-value">{html.escape(str(value))}</div>
            <div class="small-label">{html.escape(subtitle)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_model_info_card():
    st.markdown(
        f"""
        <div class="card">
            <div class="small-label">AI Sentiment Engine</div>
            <div class="big-value">{html.escape(MODEL_INFO_TITLE)}</div>
            <p class="insight-text">This product uses a fine-tuned transformer sentiment model optimized for customer feedback, reviews, and social-style text.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def detect_text_columns(df):
    common = ["text", "tweet", "review", "comment", "content", "message", "clean_text", "feedback"]
    detected = [col for col in df.columns if col.lower().strip() in common]
    text_like = [col for col in df.columns if df[col].dtype == "object" and col not in detected]
    return detected + text_like


def dataframe_to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")


def explain_keywords(text):
    tokens = re.findall(r"\b[a-zA-Z']+\b", safe_text(text).lower())
    positive = [word for word in tokens if word in POSITIVE_WORDS]
    negative = [word for word in tokens if word in NEGATIVE_WORDS]
    neutral = [word for word in tokens if word not in POSITIVE_WORDS and word not in NEGATIVE_WORDS]

    highlighted = []
    for raw_word in safe_text(text).split():
        cleaned = re.sub(r"[^a-zA-Z']", "", raw_word).lower()
        escaped = html.escape(raw_word)
        if cleaned in POSITIVE_WORDS:
            highlighted.append(f'<span class="highlight-pos">{escaped}</span>')
        elif cleaned in NEGATIVE_WORDS:
            highlighted.append(f'<span class="highlight-neg">{escaped}</span>')
        else:
            highlighted.append(f'<span class="highlight-neutral">{escaped}</span>')
    return " ".join(highlighted), positive, negative, neutral


def run_ai_prediction(text):
    try:
        result = predict_sentiment(text)
        return {
            "sentiment": result["label"],
            "confidence": result["confidence"],
            "probabilities": result["probabilities"],
            "processing_time": result["inference_time"],
            "engine": result.get("display_name", MODEL_INFO_TITLE),
            "model_source": result["model_name"],
            "error": None,
        }
    except (ValueError, RuntimeError) as error:
        return {
            "sentiment": "Unavailable",
            "confidence": 0.0,
            "probabilities": {"negative": 0.0, "neutral": 0.0, "positive": 0.0},
            "processing_time": 0.0,
            "engine": MODEL_INFO_TITLE,
            "model_source": MODEL_ID,
            "error": str(error),
        }


def run_batch(df, text_column):
    rows = []
    progress = st.progress(0)
    status = st.empty()
    total = len(df)

    for index, row in df.iterrows():
        text = safe_text(row[text_column])
        status.info(f"Analyzing feedback record {index + 1} of {total}…")
        prediction = run_ai_prediction(text)
        probabilities = prediction["probabilities"]
        highest_cat = highest_probability_category(probabilities)
        rows.append(
            {
                "original_text": text,
                "final_sentiment": prediction["sentiment"],
                "confidence": prediction["confidence"],
                "negative_probability": probabilities["negative"],
                "neutral_probability": probabilities["neutral"],
                "positive_probability": probabilities["positive"],
                "highest_probability_category": highest_cat,
            }
        )
        progress.progress((index + 1) / total)

    status.empty()
    results = pd.DataFrame(rows)
    results.to_csv(EXPORT_DIR / "sentiment_results.csv", index=False)
    return results


def style_chart(fig, theme="Dark Mode"):
    palette = get_theme_palette(theme)
    if theme == "Light Mode":
        fig.update_layout(
            paper_bgcolor=palette["bg"],
            plot_bgcolor=palette["panel_soft"],
            font=dict(color=palette["text"], size=11),
            title=dict(font=dict(color=palette["text"], size=14)),
            legend=dict(font=dict(color=palette["text"], size=11)),
            margin=dict(l=20, r=20, t=55, b=20),
        )
        fig.update_xaxes(
            gridcolor=palette["border"],
            color=palette["text"],
            tickfont=dict(color=palette["text"]),
            title_font=dict(color=palette["text"]),
        )
        fig.update_yaxes(
            gridcolor=palette["border"],
            color=palette["text"],
            tickfont=dict(color=palette["text"]),
            title_font=dict(color=palette["text"]),
        )
    else:
        fig.update_layout(
            paper_bgcolor="#07111f",
            plot_bgcolor="#0f1b2d",
            font=dict(color="#f8fafc", size=11),
            title=dict(font=dict(color="#f8fafc", size=14)),
            legend=dict(font=dict(color="#f8fafc", size=11)),
            margin=dict(l=20, r=20, t=55, b=20),
        )
        fig.update_xaxes(
            gridcolor="#263a59",
            color="#f8fafc",
            tickfont=dict(color="#f8fafc"),
            title_font=dict(color="#f8fafc"),
        )
        fig.update_yaxes(
            gridcolor="#263a59",
            color="#f8fafc",
            tickfont=dict(color="#f8fafc"),
            title_font=dict(color="#f8fafc"),
        )
    return fig


def sidebar():
    pages = ["Overview", "Analyze Text", "Batch Analysis", "Insights Dashboard", "About"]
    if "page" not in st.session_state:
        st.session_state.page = "Overview"
    if "next_page" in st.session_state:
        st.session_state.page = st.session_state.pop("next_page")
    
    st.sidebar.markdown("# 💬 SentiScope AI")
    st.sidebar.caption(PRODUCT_POSITIONING)
    
    # Theme toggle - use initial value from session state
    selected_theme = st.sidebar.radio(
        "Theme", 
        ["Light Mode", "Dark Mode"], 
        index=0 if st.session_state.theme == "Light Mode" else 1
    )
    # Only update if changed (to avoid unnecessary reloads)
    if selected_theme != st.session_state.theme:
        st.session_state.theme = selected_theme
        st.rerun()
    
    # Navigation with proper state binding
    selected = st.sidebar.radio("Navigation", pages, index=pages.index(st.session_state.page))
    
    # Only update state if selection actually changed (prevents double-click bug)
    if selected != st.session_state.page:
        st.session_state.page = selected
        st.rerun()
    
    st.sidebar.markdown("---")
    st.sidebar.info("Analyze reviews, comments, support tickets, and survey feedback with a fine-tuned transformer sentiment engine.")
    return st.session_state.page


def overview_page():
    st.markdown(
        """
            <div class="hero">
                <div class="hero-title">SentiScope AI Studio</div>
                <div class="subtitle">AI-powered customer feedback and review sentiment analytics studio for product teams, service teams, and small businesses.</div>
            </div>
        """,
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        render_card("Analyze", "Single Text", "Test a review, tweet, or support comment instantly.")
    with c2:
        render_card("Process", "Batch CSV", "Upload feedback datasets and generate predictions.")
    with c3:
        render_card("Understand", "Dashboard", "Explore sentiment trends and confidence insights.")

    st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)
    render_model_info_card()

    st.markdown('<div class="section-title">What This Product Does</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="card">
        SentiScope AI turns unstructured customer feedback into sentiment insights. It supports fast single-text analysis, CSV-based batch processing, keyword-level explanations, and downloadable results for client reporting or internal decision-making.
        </div>
        """,
        unsafe_allow_html=True,
    )


def single_analysis_page():
    st.markdown('<div class="section-title">Analyze Text</div>', unsafe_allow_html=True)
    st.caption("Analyze a single customer review, comment, tweet, or support message.")
    render_model_info_card()

    st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)
    cols = st.columns(4)
    for idx, (label, text) in enumerate(SAMPLE_TEXTS.items()):
        with cols[idx]:
            if st.button(label):
                st.session_state.single_text = text

    user_text = st.text_area(
        "Customer feedback",
        key="single_text",
        height=150,
        placeholder="Enter a product review, app comment, survey response, or support message...",
    )

    if st.button("Analyze Sentiment"):
        if not user_text.strip():
            st.info("Enter a review, comment, or feedback message to analyze.")
            return

        with st.status("Loading AI sentiment engine…", expanded=True) as status:
            st.write("Preparing text for analysis...")
            prediction = run_ai_prediction(user_text)
            st.write("Calculating confidence score...")
            status.update(label="Analysis completed successfully.", state="complete", expanded=False)

        if prediction.get("error"):
            st.error(prediction["error"])
            return

        sentiment = prediction["sentiment"]
        card_class = sentiment_class(sentiment)
        st.markdown(
            f"""
            <div class="card {card_class}">
                <div class="small-label">Final Sentiment</div>
                <div class="big-value">{html.escape(sentiment)}</div>
                <p>Confidence: <strong>{format_confidence(prediction['confidence'])}</strong></p>
                <p class="insight-text"><strong>Insight:</strong> {sentiment_insight(sentiment)}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="section-title">Probability Breakdown</div>', unsafe_allow_html=True)
        render_probability_bars(prediction["probabilities"])

        st.markdown('<div class="section-title">Keyword-Level Explanation</div>', unsafe_allow_html=True)
        highlighted, positive, negative, neutral = explain_keywords(user_text)
        st.markdown(f'<div class="card">{highlighted}</div>', unsafe_allow_html=True)
        k1, k2, k3 = st.columns(3)
        k1.info("Positive keywords: " + (", ".join(sorted(set(positive))) if positive else "None found"))
        k2.error("Negative keywords: " + (", ".join(sorted(set(negative))) if negative else "None found"))
        k3.warning("Neutral words: " + str(len(neutral)))
        st.caption("Keywords provide context clues. The final prediction is generated by the AI sentiment engine, not keyword matching alone.")


def batch_page():
    st.markdown('<div class="section-title">Batch Analysis</div>', unsafe_allow_html=True)
    st.caption("Upload a CSV file containing customer reviews, comments, or survey feedback.")
    render_model_info_card()

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is None:
        st.info("Upload a CSV file containing customer feedback to begin batch analysis.")
        return

    try:
        df = pd.read_csv(uploaded)
    except Exception as error:
        st.error(f"Could not read CSV file: {error}")
        return

    if df.empty:
        st.warning("The uploaded CSV file is empty. Please upload a file with at least one row of feedback.")
        return

    text_columns = detect_text_columns(df)
    if not text_columns:
        st.error("No feedback text column found. Please upload a CSV with columns like: text, review, comment, feedback, content, or message.")
        return

    st.markdown("**Select the column that contains the review, comment, tweet, or feedback text you want to analyze.**")
    
    # Auto-detected or user selection
    if len(text_columns) == 1:
        selected_column = text_columns[0]
        st.success(f"✓ Auto-detected text column: **{selected_column}**")
    else:
        selected_column = st.selectbox("Text column", text_columns, help="Choose the column with customer feedback text.")
    
    # Show preview of selected column
    st.markdown("**Preview of selected text column:**")
    sample_texts = df[selected_column].dropna().head(3).values
    for i, sample in enumerate(sample_texts, 1):
        st.caption(f"Sample {i}: {safe_text(sample)[:100]}…" if len(safe_text(sample)) > 100 else f"Sample {i}: {sample}")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Rows", len(df))
    c2.metric("Missing Text", int(df[selected_column].isna().sum()))
    c3.metric("Columns", len(df.columns))

    st.markdown("**Dataset Preview**")
    st.dataframe(df.head(8), use_container_width=True)

    if st.button("Run Feedback Sentiment Analysis"):
        clean_df = df.dropna(subset=[selected_column]).copy()
        if clean_df.empty:
            st.warning("No feedback text rows found after removing missing values.")
            return
        with st.spinner("Analyzing feedback with AI. This may take a moment."):
            results = run_batch(clean_df, selected_column)
        st.session_state.batch_results = results
        st.success("✓ Analysis completed successfully.")

    results = st.session_state.get("batch_results")
    if results is not None and not results.empty:
        st.markdown('<div class="section-title">Prediction Results</div>', unsafe_allow_html=True)
        
        # Display with badge for highest probability category
        display_results = results.copy()
        display_results.columns = [
            "Text", "Sentiment", "Confidence", "Negative %", "Neutral %", "Positive %", "Category"
        ]
        st.dataframe(display_results, use_container_width=True)
        
        # Download clean CSV
        export_df = clean_export_dataframe(results)
        st.download_button(
            "⬇ Download Results CSV",
            data=dataframe_to_csv_bytes(export_df),
            file_name="sentiscope_analysis_results.csv",
            mime="text/csv",
        )
        if st.button("View Insights Dashboard"):
            st.session_state.next_page = "Insights Dashboard"
            st.session_state.scroll_to_top = True
            st.rerun()


def analytics_page():
    # Scroll to top if navigated from batch page
    if st.session_state.get("scroll_to_top"):
        scroll_to_top()
        st.session_state.scroll_to_top = False
    
    st.markdown('<div class="section-title">Customer Sentiment Overview</div>', unsafe_allow_html=True)
    render_model_info_card()
    
    results = st.session_state.get("batch_results")
    if results is None or results.empty:
        st.info("No analyzed feedback results found yet. Upload a CSV in Batch Analysis and run sentiment analysis to see dashboard insights.")
        return

    # Recalculate using the new column names from run_batch
    sentiment_counts = results["final_sentiment"].value_counts()
    avg_conf = results["confidence"].mean()
    most_common = sentiment_counts.idxmax() if not sentiment_counts.empty else "N/A"
    
    total_feedback = len(results)
    positive_count = int(sentiment_counts.get("Positive", 0))
    neutral_count = int(sentiment_counts.get("Neutral", 0))
    negative_count = int(sentiment_counts.get("Negative", 0))

    # KPI Cards
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Feedback Analyzed", total_feedback)
    k2.metric("Positive Feedback", positive_count)
    k3.metric("Neutral Feedback", neutral_count)
    k4.metric("Negative Feedback", negative_count)
    
    k5, k6 = st.columns(2)
    k5.metric("Average Confidence", format_confidence(avg_conf))
    k6.metric("Dominant Sentiment", most_common)

    # Charts
    st.markdown('<div class="section-title">Feedback Mood Breakdown</div>', unsafe_allow_html=True)
    chart_df = sentiment_counts.reset_index()
    chart_df.columns = ["Sentiment", "Count"]
    colors = {"Positive": "#22c55e", "Negative": "#ef4444", "Neutral": "#60a5fa"}
    
    theme = st.session_state.get("theme", "Dark Mode")

    c1, c2 = st.columns(2)
    with c1:
        fig = px.pie(chart_df, names="Sentiment", values="Count", title="Sentiment Distribution", color="Sentiment", color_discrete_map=colors)
        fig.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(style_chart(fig, theme), use_container_width=True)
    with c2:
        fig = px.bar(chart_df, x="Sentiment", y="Count", title="Sentiment Counts", color="Sentiment", color_discrete_map=colors, text="Count")
        st.plotly_chart(style_chart(fig, theme), use_container_width=True)

    st.markdown('<div class="section-title">Confidence Distribution</div>', unsafe_allow_html=True)
    fig = px.histogram(results, x="confidence", nbins=12, title="Model Confidence Scores", color="final_sentiment", color_discrete_map=colors)
    fig.update_xaxes(tickformat=".0%", title="Confidence")
    st.plotly_chart(style_chart(fig, theme), use_container_width=True)

    st.markdown('<div class="section-title">Key Sentiment Insights</div>', unsafe_allow_html=True)
    st.dataframe(results.head(25), use_container_width=True)


def about_page():
    st.markdown('<div class="section-title">About SentiScope AI Studio</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="card">
        SentiScope AI Studio is a client-ready sentiment analytics dashboard for customer feedback. It helps teams understand how users feel about products, services, and support experiences. Upload reviews, comments, or survey responses, and get instant sentiment insights with exportable results for business reporting.
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    render_model_info_card()
    
    st.markdown('<div class="section-title">Technical Details</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="card">
        <div class="small-label">AI Sentiment Engine</div>
        <p>Fine-tuned Twitter-RoBERTa sentiment transformer, optimized for customer feedback, reviews, and social-style text.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    st.markdown('<div class="section-title">Balanced Quality Score</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="card">
        <div class="big-value">85.58%</div>
        <p class="insight-text">This score measures model performance across Positive, Neutral, and Negative sentiment classes equally. It is based on macro F1, a standard machine learning metric that treats all classes fairly. This score was measured on a held-out test set from the training dataset.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    st.markdown('<div class="section-title">Recommended Workflow</div>', unsafe_allow_html=True)
    st.markdown("1. **Test single examples** in the Analyze Text page with real feedback")
    st.markdown("2. **Upload CSV feedback** in Batch Analysis with customer reviews or support comments")
    st.markdown("3. **Review KPIs and charts** in Insights Dashboard to understand sentiment trends")
    st.markdown("4. **Download results** for client reports, dashboards, or business review")


def main():
    if "theme" not in st.session_state:
        st.session_state.theme = "Dark Mode"
    load_css(st.session_state.theme)
    page = sidebar()
    if page == "Overview":
        overview_page()
    elif page == "Analyze Text":
        single_analysis_page()
    elif page == "Batch Analysis":
        batch_page()
    elif page == "Insights Dashboard":
        analytics_page()
    elif page == "About":
        about_page()


if __name__ == "__main__":
    main()
