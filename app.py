import html
import re
import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from bert_predict import predict_distilbert_sentiment
from predict import model_files_exist, predict_sentiment
from preprocessing import NEGATIVE_WORDS, POSITIVE_WORDS, preprocess_text

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


def load_css():
    st.markdown(
        """
        <style>
        :root {
            --bg: #07111f;
            --panel: #0f1b2d;
            --panel-soft: #14243a;
            --border: #263a59;
            --text: #f8fafc;
            --muted: #a8b3c7;
            --cyan: #38bdf8;
            --green: #22c55e;
            --red: #ef4444;
            --blue: #60a5fa;
        }
        .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
            background: var(--bg) !important;
            color: var(--text) !important;
        }
        [data-testid="stSidebar"] {
            background: #091525 !important;
            border-right: 1px solid var(--border);
        }
        h1, h2, h3, h4, h5, h6, p, li, label, span, div {
            color: var(--text);
        }
        .hero {
            padding: 2rem;
            border: 1px solid var(--border);
            border-radius: 18px;
            background: linear-gradient(135deg, #0f1b2d 0%, #10263e 55%, #082235 100%);
            margin-bottom: 1.5rem;
        }
        .hero-title {
            font-size: 3rem;
            font-weight: 800;
            letter-spacing: 0;
            margin-bottom: .4rem;
        }
        .subtitle {
            color: var(--muted);
            font-size: 1.08rem;
            line-height: 1.7;
        }
        .section-title {
            font-size: 1.45rem;
            font-weight: 750;
            margin: 1.2rem 0 .7rem 0;
        }
        .card {
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 1.1rem;
            box-shadow: 0 18px 40px rgba(0,0,0,.22);
            min-height: 100px;
        }
        .metric-card {
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 1rem;
        }
        .small-label { color: var(--muted); font-size: .85rem; }
        .big-value { font-size: 1.65rem; font-weight: 800; margin-top: .25rem; }
        .positive { border-left: 5px solid var(--green); }
        .negative { border-left: 5px solid var(--red); }
        .neutral { border-left: 5px solid var(--blue); }
        .highlight-pos { background: rgba(34,197,94,.22); color: #bbf7d0; padding: 2px 6px; border-radius: 7px; }
        .highlight-neg { background: rgba(239,68,68,.24); color: #fecaca; padding: 2px 6px; border-radius: 7px; }
        .highlight-neutral { color: #cbd5e1; }
        .stButton > button, .stDownloadButton > button {
            background: linear-gradient(135deg, #0ea5e9, #2563eb) !important;
            color: white !important;
            border: 1px solid rgba(255,255,255,.18) !important;
            border-radius: 12px !important;
            font-weight: 700 !important;
            padding: .65rem 1rem !important;
        }
        .stButton > button:hover, .stDownloadButton > button:hover {
            border-color: #7dd3fc !important;
            filter: brightness(1.08);
        }
        textarea, input, [data-baseweb="select"] > div {
            background: #0b1728 !important;
            color: #f8fafc !important;
            border-color: #33506f !important;
        }
        textarea::placeholder, input::placeholder { color: #94a3b8 !important; opacity: 1 !important; }
        textarea { caret-color: #38bdf8 !important; }
        [data-testid="stDataFrame"] { border: 1px solid var(--border); border-radius: 12px; overflow: hidden; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def safe_text(value):
    return "" if value is None else str(value)


def sentiment_class(sentiment):
    label = safe_text(sentiment).lower()
    if "positive" in label:
        return "positive"
    if "negative" in label:
        return "negative"
    return "neutral"


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
    result = predict_distilbert_sentiment(text)
    if not result.get("error"):
        result["engine"] = "DistilBERT sentiment engine"
        return result

    if model_files_exist():
        fallback = predict_sentiment(text)
        fallback["engine"] = "TF-IDF fallback engine"
        fallback["error"] = None
        return fallback

    return {
        "sentiment": "Unavailable",
        "confidence": 0.0,
        "processing_time": 0.0,
        "engine": "No model available",
        "error": result.get("error", "Sentiment engine could not be loaded."),
    }


def run_batch(df, text_column):
    rows = []
    progress = st.progress(0)
    status = st.empty()
    total = len(df)

    for index, row in df.iterrows():
        text = safe_text(row[text_column])
        status.info(f"Analyzing record {index + 1} of {total}...")
        prediction = run_ai_prediction(text)
        rows.append(
            {
                "Original Text": text,
                "Sentiment": prediction["sentiment"],
                "Confidence": prediction["confidence"],
                "Processing Time": prediction["processing_time"],
                "Engine": prediction["engine"],
            }
        )
        progress.progress((index + 1) / total)

    status.success("Batch analysis completed.")
    results = pd.DataFrame(rows)
    results.to_csv(EXPORT_DIR / "sentiment_results.csv", index=False)
    return results


def style_chart(fig):
    fig.update_layout(
        paper_bgcolor="#07111f",
        plot_bgcolor="#0f1b2d",
        font=dict(color="#f8fafc"),
        legend=dict(font=dict(color="#f8fafc")),
        margin=dict(l=20, r=20, t=55, b=20),
    )
    fig.update_xaxes(gridcolor="#263a59", color="#f8fafc")
    fig.update_yaxes(gridcolor="#263a59", color="#f8fafc")
    return fig


def sidebar():
    pages = ["Overview", "Single Analysis", "Batch CSV Analysis", "Analytics Dashboard", "About"]
    if "page" not in st.session_state:
        st.session_state.page = "Overview"
    if "next_page" in st.session_state:
        st.session_state.page = st.session_state.pop("next_page")
    st.sidebar.markdown("# 💬 SentiScope AI")
    st.sidebar.caption("Customer feedback sentiment analytics studio")
    selected = st.sidebar.radio("Navigation", pages, index=pages.index(st.session_state.page))
    st.session_state.page = selected
    st.sidebar.markdown("---")
    st.sidebar.info("Analyze reviews, comments, support tickets, and survey feedback with an AI-powered dashboard.")
    return selected


def overview_page():
    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">SentiScope AI Studio</div>
            <div class="subtitle">AI-powered customer feedback and review sentiment analytics for product teams, service teams, and small businesses.</div>
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
    st.markdown('<div class="section-title">Single Text Sentiment Analysis</div>', unsafe_allow_html=True)
    st.caption("Analyze one customer review, comment, tweet, or support message.")

    cols = st.columns(4)
    for idx, (label, text) in enumerate(SAMPLE_TEXTS.items()):
        with cols[idx]:
            if st.button(label):
                st.session_state.single_text = text

    user_text = st.text_area(
        "Customer feedback text",
        key="single_text",
        height=150,
        placeholder="Paste a product review, app comment, survey response, or customer support message...",
    )

    if st.button("Analyze Sentiment"):
        if not user_text.strip():
            st.warning("Please enter text before running sentiment analysis.")
            return

        with st.status("Analyzing customer feedback...", expanded=True) as status:
            st.write("Running AI sentiment engine...")
            prediction = run_ai_prediction(user_text)
            st.write("Preparing keyword explanation and preprocessing view...")
            steps = preprocess_text(user_text)
            status.update(label="Analysis complete.", state="complete", expanded=False)

        if prediction.get("error"):
            st.error(prediction["error"])
            return

        sentiment = prediction["sentiment"]
        card_class = sentiment_class(sentiment)
        st.markdown(
            f"""
            <div class="card {card_class}">
                <div class="small-label">Sentiment Result</div>
                <div class="big-value">{html.escape(sentiment)}</div>
                <p>Confidence: <strong>{prediction['confidence']:.2f}%</strong></p>
                <p>Processing Time: <strong>{prediction['processing_time']:.4f}s</strong></p>
                <p class="small-label">Engine: {html.escape(prediction['engine'])}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="section-title">Keyword-Level Explanation</div>', unsafe_allow_html=True)
        highlighted, positive, negative, neutral = explain_keywords(user_text)
        st.markdown(f'<div class="card">{highlighted}</div>', unsafe_allow_html=True)
        k1, k2, k3 = st.columns(3)
        k1.info("Positive keywords: " + (", ".join(sorted(set(positive))) if positive else "None found"))
        k2.error("Negative keywords: " + (", ".join(sorted(set(negative))) if negative else "None found"))
        k3.warning("Neutral/unknown words: " + str(len(neutral)))
        st.caption("This explanation uses keyword-level sentiment clues. The final prediction is generated by the sentiment model, not by keyword matching alone.")

        st.markdown('<div class="section-title">Preprocessing Pipeline</div>', unsafe_allow_html=True)
        p1, p2 = st.columns(2)
        with p1:
            st.markdown("**Original Text**")
            st.write(steps["original_text"])
            st.markdown("**Lowercased Text**")
            st.write(steps["lowercased_text"])
            st.markdown("**Cleaned Text**")
            st.write(steps["cleaned_text"])
        with p2:
            st.markdown("**Tokens**")
            st.write(steps["raw_tokens"])
            st.markdown("**After Stopword Removal**")
            st.write(steps["tokens"])
            st.markdown("**Final Processed Text**")
            st.write(steps["final_text"])


def batch_page():
    st.markdown('<div class="section-title">Batch CSV Analysis</div>', unsafe_allow_html=True)
    st.caption("Upload a CSV file containing customer reviews, comments, tickets, or survey feedback.")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is None:
        st.info("Upload a CSV file to begin batch sentiment analysis.")
        return

    try:
        df = pd.read_csv(uploaded)
    except Exception as error:
        st.error(f"Could not read CSV file: {error}")
        return

    if df.empty:
        st.warning("The uploaded CSV file is empty.")
        return

    text_columns = detect_text_columns(df)
    if not text_columns:
        st.error("No text-like column was found. Please upload a CSV containing review, comment, content, or feedback text.")
        return

    st.markdown("**Select the column that contains customer feedback text.**")
    selected_column = st.selectbox("Text column", text_columns)

    st.success(f"Using text column: {selected_column}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", len(df))
    c2.metric("Missing Text", int(df[selected_column].isna().sum()))
    c3.metric("Columns", len(df.columns))

    st.markdown("**Dataset Preview**")
    st.dataframe(df.head(8), use_container_width=True)

    if st.button("Run Batch Analysis"):
        clean_df = df.dropna(subset=[selected_column]).copy()
        if clean_df.empty:
            st.warning("No usable text rows found after removing missing values.")
            return
        with st.spinner("Analyzing CSV feedback. This may take a moment for large files..."):
            results = run_batch(clean_df, selected_column)
        st.session_state.batch_results = results
        st.success("Batch sentiment analysis completed.")

    results = st.session_state.get("batch_results")
    if results is not None and not results.empty:
        st.markdown('<div class="section-title">Prediction Results</div>', unsafe_allow_html=True)
        st.dataframe(results, use_container_width=True)
        st.download_button(
            "⬇ Download Results CSV",
            data=dataframe_to_csv_bytes(results),
            file_name="sentiscope_results.csv",
            mime="text/csv",
        )
        if st.button("Open Analytics Dashboard"):
            st.session_state.next_page = "Analytics Dashboard"
            st.rerun()


def analytics_page():
    st.markdown('<div class="section-title">Analytics Dashboard</div>', unsafe_allow_html=True)
    results = st.session_state.get("batch_results")
    if results is None or results.empty:
        st.warning("No batch results found yet. Run Batch CSV Analysis first to populate the dashboard.")
        return

    sentiment_counts = results["Sentiment"].value_counts()
    avg_conf = results["Confidence"].mean()
    most_common = sentiment_counts.idxmax() if not sentiment_counts.empty else "N/A"

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Feedback", len(results))
    k2.metric("Positive", int(sentiment_counts.get("Positive", 0)))
    k3.metric("Negative", int(sentiment_counts.get("Negative", 0)))
    k4.metric("Avg Confidence", f"{avg_conf:.2f}%")
    st.metric("Most Common Sentiment", most_common)

    chart_df = sentiment_counts.reset_index()
    chart_df.columns = ["Sentiment", "Count"]
    colors = {"Positive": "#22c55e", "Negative": "#ef4444", "Neutral": "#60a5fa"}

    c1, c2 = st.columns(2)
    with c1:
        fig = px.pie(chart_df, names="Sentiment", values="Count", title="Sentiment Distribution", color="Sentiment", color_discrete_map=colors)
        fig.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(style_chart(fig), use_container_width=True)
    with c2:
        fig = px.bar(chart_df, x="Sentiment", y="Count", title="Sentiment Counts", color="Sentiment", color_discrete_map=colors, text="Count")
        st.plotly_chart(style_chart(fig), use_container_width=True)

    fig = px.histogram(results, x="Confidence", nbins=12, title="Confidence Distribution", color="Sentiment", color_discrete_map=colors)
    st.plotly_chart(style_chart(fig), use_container_width=True)

    st.markdown('<div class="section-title">Recent Feedback Results</div>', unsafe_allow_html=True)
    st.dataframe(results.head(25), use_container_width=True)


def about_page():
    st.markdown('<div class="section-title">About SentiScope AI Studio</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="card">
        SentiScope AI Studio is a client-ready sentiment analytics dashboard for customer feedback. It helps teams understand how users feel about products, services, applications, and support experiences. The product focuses on clear outputs, exportable results, and practical analytics rather than academic model reporting.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="section-title">Recommended Workflow</div>', unsafe_allow_html=True)
    st.write("1. Test a few examples in Single Analysis.")
    st.write("2. Upload a customer feedback CSV in Batch CSV Analysis.")
    st.write("3. Review sentiment KPIs and charts in Analytics Dashboard.")
    st.write("4. Download the CSV results for reporting or business review.")


def main():
    load_css()
    page = sidebar()
    if page == "Overview":
        overview_page()
    elif page == "Single Analysis":
        single_analysis_page()
    elif page == "Batch CSV Analysis":
        batch_page()
    elif page == "Analytics Dashboard":
        analytics_page()
    elif page == "About":
        about_page()


if __name__ == "__main__":
    main()
