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
TEXT_COLUMN_PRIORITY = (
    "text",
    "review",
    "tweet_text",
    "full_text",
    "comment",
    "feedback",
    "tweet",
    "content",
    "message",
    "clean_text",
)

PAGE_ICONS = {
    "Overview": "🏠",
    "Analyze Text": "🔍",
    "Batch Analysis": "📊",
    "Insights Dashboard": "📈",
    "About": "ℹ️",
}


def get_theme_palette(theme_name):
    if theme_name == "Light Mode":
        return {
            "bg": "#F4F6FF",
            "sidebar": "#FFFFFF",
            "panel": "#FFFFFF",
            "panel_soft": "#EEF2FF",
            "border": "#C7D2FE",
            "text": "#0F172A",
            "muted": "#475569",
            "accent": "#6366F1",
            "accent_2": "#A855F7",
            "green": "#16A34A",
            "red": "#DC2626",
            "blue": "#3B82F6",
            "shadow": "0 20px 40px rgba(99,102,241,.13)",
            "input_bg": "#FFFFFF",
            "grid": "#C7D2FE",
            "glow_green": "rgba(22,163,74,.25)",
            "glow_red": "rgba(220,38,38,.25)",
            "glow_blue": "rgba(59,130,246,.25)",
            "glow_accent": "rgba(99,102,241,.30)",
            "hero_grad": "linear-gradient(135deg, #EEF2FF 0%, #E0E7FF 50%, #F4F6FF 100%)",
        }
    return {
        "bg": "#060B18",
        "sidebar": "#080E1C",
        "panel": "#0D1526",
        "panel_soft": "#111E33",
        "border": "#1A2847",
        "text": "#F0F6FF",
        "muted": "#94A3B8",
        "accent": "#818CF8",
        "accent_2": "#A78BFA",
        "green": "#34D399",
        "red": "#F87171",
        "blue": "#60A5FA",
        "shadow": "0 24px 48px rgba(0,0,0,.35)",
        "input_bg": "#0A1020",
        "grid": "#1A2847",
        "glow_green": "rgba(52,211,153,.20)",
        "glow_red": "rgba(248,113,113,.20)",
        "glow_blue": "rgba(96,165,250,.20)",
        "glow_accent": "rgba(129,140,248,.20)",
        "hero_grad": "linear-gradient(135deg, #0D1526 0%, #111E33 50%, #0D1526 100%)",
    }


def load_css(theme_name):
    palette = get_theme_palette(theme_name)
    css = f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Space+Grotesk:wght@400;500;600;700&display=swap');

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
            --glow-green: {palette["glow_green"]};
            --glow-red: {palette["glow_red"]};
            --glow-blue: {palette["glow_blue"]};
            --glow-accent: {palette["glow_accent"]};
        }}

        /* ── Base & Fluid Layout Constraints ── */
        *, *::before, *::after {{ box-sizing: border-box; }}
        html, body, .stApp {{
            font-family: 'Inter', 'Segoe UI', system-ui, sans-serif !important;
            overflow-x: hidden !important;
        }}
        .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {{
            background: var(--bg) !important;
            color: var(--text) !important;
        }}
        [data-testid="stMain"] > div {{
            max-width: 1440px !important;
            margin-left: auto !important;
            margin-right: auto !important;
            padding-bottom: env(safe-area-inset-bottom, 1.5rem) !important;
        }}
        [data-testid="stSidebar"] {{
            background: var(--sidebar) !important;
            border-right: 1px solid var(--border) !important;
        }}
        h1, h2, h3, h4, h5, h6 {{
            font-family: 'Space Grotesk', 'Inter', sans-serif !important;
            color: var(--text) !important;
            letter-spacing: -0.02em;
        }}
        p, li, label, span, div, caption {{
            color: var(--text);
        }}

        /* ── Mobile Sidebar Menu Toggle Button (Header / Collapsed Control) ── */
        [data-testid="stHeader"] {{
            background: transparent !important;
            z-index: 99999 !important;
        }}
        [data-testid="collapsedControl"],
        [data-testid="stSidebarCollapseButton"],
        [data-testid="stHeader"] button,
        [data-testid="stSidebarToggle"],
        button[kind="header"] {{
            background: linear-gradient(135deg, var(--accent), var(--accent-2)) !important;
            color: #FFFFFF !important;
            border: 1px solid rgba(255, 255, 255, 0.3) !important;
            border-radius: 12px !important;
            min-width: 44px !important;
            min-height: 44px !important;
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
            box-shadow: 0 4px 14px var(--glow-accent) !important;
            margin: 0.5rem !important;
            transition: all 0.22s ease !important;
            opacity: 1 !important;
            visibility: visible !important;
        }}
        [data-testid="collapsedControl"] *,
        [data-testid="stSidebarCollapseButton"] *,
        [data-testid="stHeader"] button *,
        [data-testid="stSidebarToggle"] *,
        button[kind="header"] * {{
            fill: #FFFFFF !important;
            stroke: #FFFFFF !important;
            color: #FFFFFF !important;
        }}
        [data-testid="collapsedControl"] svg,
        [data-testid="stSidebarCollapseButton"] svg,
        [data-testid="stHeader"] button svg,
        [data-testid="stSidebarToggle"] svg,
        button[kind="header"] svg {{
            width: 22px !important;
            height: 22px !important;
        }}
        [data-testid="collapsedControl"]:hover,
        [data-testid="stSidebarCollapseButton"]:hover,
        [data-testid="stHeader"] button:hover {{
            filter: brightness(1.15) !important;
            transform: scale(1.05) !important;
            box-shadow: 0 6px 18px var(--glow-accent) !important;
            border-color: rgba(255, 255, 255, 0.5) !important;
        }}

        /* ── Safe Area Insets for Mobile / Modern Devices ── */
        @supports (padding: env(safe-area-inset-top)) {{
            [data-testid="stAppViewContainer"] {{
                padding-top: env(safe-area-inset-top, 0px) !important;
                padding-bottom: env(safe-area-inset-bottom, 0px) !important;
                padding-left: env(safe-area-inset-left, 0px) !important;
                padding-right: env(safe-area-inset-right, 0px) !important;
            }}
        }}

        /* ── Sidebar Brand ── */
        .sidebar-brand {{
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 0 0 0.25rem 0;
            margin-bottom: 0.5rem;
        }}
        .sidebar-logo-ring {{
            width: 38px;
            height: 38px;
            border-radius: 10px;
            background: linear-gradient(135deg, var(--accent), var(--accent-2));
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.1rem;
            box-shadow: 0 4px 14px var(--glow-accent);
            flex-shrink: 0;
        }}
        .sidebar-brand-name {{
            font-family: 'Space Grotesk', sans-serif;
            font-weight: 700;
            font-size: 1.05rem;
            color: var(--text) !important;
            line-height: 1.2;
        }}
        .sidebar-brand-sub {{
            font-size: 0.7rem;
            color: var(--muted);
            font-weight: 400;
        }}
        .engine-badge {{
            display: inline-flex;
            align-items: center;
            gap: 5px;
            background: var(--panel-soft);
            border: 1px solid var(--border);
            border-radius: 999px;
            padding: 4px 10px;
            font-size: 0.72rem;
            color: var(--green);
            font-weight: 600;
            margin-bottom: 0.75rem;
        }}
        .engine-badge-dot {{
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background: var(--green);
            box-shadow: 0 0 6px var(--glow-green);
            animation: pulse-dot 2s ease-in-out infinite;
        }}
        @keyframes pulse-dot {{
            0%, 100% {{ opacity: 1; transform: scale(1); }}
            50% {{ opacity: 0.6; transform: scale(0.8); }}
        }}

        /* ── Fluid Hero ── */
        @keyframes hero-shimmer {{
            0% {{ background-position: 0% 50%; }}
            50% {{ background-position: 100% 50%; }}
            100% {{ background-position: 0% 50%; }}
        }}
        @keyframes fadeInUp {{
            from {{ opacity: 0; transform: translateY(18px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        @keyframes fadeIn {{
            from {{ opacity: 0; }}
            to {{ opacity: 1; }}
        }}
        .hero {{
            padding: clamp(1.5rem, 4vw, 2.5rem);
            border: 1px solid var(--border);
            border-radius: 24px;
            background: {palette["hero_grad"]};
            background-size: 300% 300%;
            animation: hero-shimmer 8s ease infinite, fadeInUp 0.6s ease forwards;
            margin-bottom: 1.75rem;
            position: relative;
            overflow: hidden;
        }}
        .hero::before {{
            content: '';
            position: absolute;
            top: -60px; right: -60px;
            width: 220px; height: 220px;
            border-radius: 50%;
            background: radial-gradient(circle, var(--glow-accent) 0%, transparent 70%);
            pointer-events: none;
        }}
        .hero::after {{
            content: '';
            position: absolute;
            bottom: -40px; left: 20%;
            width: 160px; height: 160px;
            border-radius: 50%;
            background: radial-gradient(circle, var(--glow-blue) 0%, transparent 70%);
            pointer-events: none;
        }}
        .hero-eyebrow {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: linear-gradient(135deg, var(--accent), var(--accent-2));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            font-weight: 700;
            font-size: 0.78rem;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            margin-bottom: 0.6rem;
        }}
        .hero-title {{
            font-family: 'Space Grotesk', sans-serif;
            font-size: clamp(2rem, 5vw, 3.2rem) !important;
            font-weight: 800;
            letter-spacing: -0.03em;
            line-height: 1.1;
            margin-bottom: 0.75rem;
            color: var(--text) !important;
        }}
        .hero-title .accent-word {{
            background: linear-gradient(135deg, var(--accent), var(--accent-2));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        .subtitle {{
            color: var(--muted);
            font-size: 1.05rem;
            line-height: 1.75;
            max-width: 560px;
            margin-bottom: 1.5rem;
        }}
        .hero-stats {{
            display: flex;
            gap: 1.5rem;
            flex-wrap: wrap;
        }}
        .hero-stat {{
            display: flex;
            align-items: center;
            gap: 7px;
            font-size: 0.85rem;
            font-weight: 600;
            color: var(--muted);
        }}
        .hero-stat-icon {{
            font-size: 1rem;
        }}
        .hero-stat strong {{
            color: var(--text);
        }}

        /* ── Section Titles ── */
        .section-title {{
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.35rem;
            font-weight: 700;
            margin: 1.5rem 0 0.75rem 0;
            color: var(--text) !important;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .section-title::after {{
            content: '';
            flex: 1;
            height: 1px;
            background: linear-gradient(to right, var(--border), transparent);
            margin-left: 0.5rem;
        }}
        .section-spacer {{ height: 1.25rem; }}

        /* ── Cards (Mobile-First Controlled Sizing & Fluid Padding) ── */
        .card {{
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: clamp(1rem, 3vw, 1.4rem); /* Adaptive internal padding across viewports */
            box-shadow: var(--shadow);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            transition: box-shadow 0.25s ease, transform 0.25s ease, border-color 0.25s ease;
            animation: fadeInUp 0.45s ease forwards;
            width: 100%; /* Ensures card fills parent container smoothly */
            max-width: 100%; /* Prevents card overflow on mobile */
        }}
        .card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 28px 56px rgba(0,0,0,.28);
            border-color: var(--accent);
        }}
        .card.positive {{
            border-left: 4px solid var(--green);
            box-shadow: var(--shadow), -2px 0 20px var(--glow-green);
        }}
        .card.negative {{
            border-left: 4px solid var(--red);
            box-shadow: var(--shadow), -2px 0 20px var(--glow-red);
        }}
        .card.neutral {{
            border-left: 4px solid var(--blue);
            box-shadow: var(--shadow), -2px 0 20px var(--glow-blue);
        }}

        /* ── Nav Feature Cards (Mobile-First Strict Sizing & Anti-Cropping) ── */
        .nav-card {{
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: clamp(1rem, 2.5vw, 1.4rem); /* Adaptive spacing prevents text clipping */
            box-shadow: var(--shadow);
            transition: all 0.25s cubic-bezier(0.34, 1.56, 0.64, 1);
            text-align: left;
            width: 100%; /* Prevents card squishing on mobile */
            min-width: 0; /* Ensures flex container child text truncation doesn't break layout */
            min-height: 220px; /* Fluid minimum height prevents text cropping on 1024px tablet screens */
            height: 100%; /* Dynamic height matching across multi-column layout */
            display: flex;
            flex-direction: column;
            justify-content: flex-start;
            margin-bottom: clamp(0.85rem, 2vw, 1.25rem); /* Space between card and CTA button below */
            position: relative;
            overflow: visible; /* Prevents text from getting cut off at bottom of card */
        }}
        .nav-card:hover {{
            transform: translateY(-5px);
            border-color: var(--accent);
            box-shadow: 0 24px 48px rgba(0,0,0,.30), 0 0 0 1px var(--accent);
        }}
        .nav-card-icon {{
            font-size: clamp(1.5rem, 3vw, 2rem); /* Dynamic scaling for card icon */
            margin-bottom: 0.6rem;
            display: block;
        }}
        .nav-card-label {{
            font-size: clamp(0.68rem, 1.8vw, 0.72rem);
            font-weight: 700;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            color: var(--accent);
            margin-bottom: 0.2rem;
        }}
        .nav-card-title {{
            font-family: 'Space Grotesk', sans-serif;
            font-size: clamp(1rem, 2.2vw, 1.15rem); /* Fluid title scaling */
            font-weight: 700;
            color: var(--text);
            margin-bottom: 0.4rem;
        }}
        .nav-card-desc {{
            font-size: clamp(0.78rem, 1.8vw, 0.82rem);
            color: var(--muted);
            line-height: 1.55;
            flex: 1;
        }}

        @media (max-width: 1150px) {{
            .nav-card {{
                min-height: 210px !important;
                padding: 1.1rem 1rem !important;
                margin-bottom: 0.85rem !important;
            }}
            .nav-card-title {{
                font-size: 1.05rem !important;
            }}
            .nav-card-desc {{
                font-size: 0.8rem !important;
                line-height: 1.5 !important;
            }}
        }}

        @media (max-width: 768px) {{
            .nav-card {{
                height: auto !important; /* Fluid height adaptation for small screens */
                min-height: 170px !important; /* Strict minimum height constraint on mobile */
                margin-bottom: 0.85rem !important; /* Reduced bottom margin on small screens */
            }}
        }}

        /* ── Metric Cards (Mobile-First Layout) ── */
        .kpi-card {{
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: clamp(0.85rem, 2.5vw, 1.25rem); /* Adaptive padding */
            box-shadow: var(--shadow);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
            animation: fadeInUp 0.5s ease forwards;
            width: 100%; /* Prevents card squishing on mobile */
        }}
        .kpi-card:hover {{
            transform: translateY(-3px);
        }}
        .kpi-card.green {{ border-top: 3px solid var(--green); box-shadow: var(--shadow), 0 -2px 16px var(--glow-green); }}
        .kpi-card.red   {{ border-top: 3px solid var(--red);   box-shadow: var(--shadow), 0 -2px 16px var(--glow-red);   }}
        .kpi-card.blue  {{ border-top: 3px solid var(--blue);  box-shadow: var(--shadow), 0 -2px 16px var(--glow-blue);  }}
        .kpi-card.accent{{ border-top: 3px solid var(--accent);box-shadow: var(--shadow), 0 -2px 16px var(--glow-accent);}}
        .kpi-icon {{ font-size: clamp(1.2rem, 3vw, 1.5rem); margin-bottom: 0.4rem; }}
        .kpi-label {{ font-size: clamp(0.68rem, 1.8vw, 0.75rem); font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); margin-bottom: 0.25rem; }}
        .kpi-value {{ font-family: 'Space Grotesk', sans-serif; font-size: clamp(1.4rem, 3.5vw, 2rem); font-weight: 800; color: var(--text); line-height: 1; }} /* Fluid KPI metric font */
        .kpi-sub   {{ font-size: 0.75rem; color: var(--muted); margin-top: 0.25rem; }}

        /* ── Model Info Card (Adaptive Mobile Layout) ── */
        .model-card {{
            background: linear-gradient(135deg, var(--panel) 0%, var(--panel-soft) 100%);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: clamp(0.85rem, 2.5vw, 1.1rem) clamp(1rem, 3vw, 1.4rem);
            display: flex;
            align-items: center;
            gap: clamp(0.6rem, 2vw, 1rem);
            box-shadow: var(--shadow);
            margin-bottom: 1.25rem;
            flex-wrap: wrap; /* Prevents overflow on narrow screens */
        }}
        .model-card-icon {{
            width: clamp(38px, 8vw, 44px); height: clamp(38px, 8vw, 44px);
            border-radius: 12px;
            background: linear-gradient(135deg, var(--accent), var(--accent-2));
            display: flex; align-items: center; justify-content: center;
            font-size: clamp(1.1rem, 3vw, 1.3rem);
            box-shadow: 0 4px 16px var(--glow-accent);
            flex-shrink: 0;
        }}
        .model-card-label {{ font-size: 0.72rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: var(--accent); }}
        .model-card-title {{ font-family: 'Space Grotesk', sans-serif; font-size: clamp(0.9rem, 2vw, 1rem); font-weight: 700; color: var(--text); margin: 2px 0; }}
        .model-card-sub {{ font-size: 0.78rem; color: var(--muted); }}
        .model-score-pill {{
            margin-left: auto;
            background: var(--panel-soft);
            border: 1px solid var(--border);
            border-radius: 999px;
            padding: 5px 14px;
            font-size: 0.78rem;
            font-weight: 700;
            color: var(--green);
            white-space: nowrap;
        }}

        /* ── Sentiment Result Card (Fluid Sizing) ── */
        .result-card {{
            border-radius: 20px;
            padding: clamp(1.25rem, 4vw, 2rem); /* Dynamic padding for mobile/desktop */
            position: relative;
            overflow: hidden;
            animation: fadeInUp 0.5s ease forwards;
            width: 100%; /* Prevents card squishing on mobile */
        }}
        .result-card.positive {{
            background: linear-gradient(135deg, var(--panel) 0%, rgba(52,211,153,0.08) 100%);
            border: 1px solid var(--green);
            box-shadow: 0 20px 48px var(--glow-green), var(--shadow);
        }}
        .result-card.negative {{
            background: linear-gradient(135deg, var(--panel) 0%, rgba(248,113,113,0.08) 100%);
            border: 1px solid var(--red);
            box-shadow: 0 20px 48px var(--glow-red), var(--shadow);
        }}
        .result-card.neutral {{
            background: linear-gradient(135deg, var(--panel) 0%, rgba(96,165,250,0.08) 100%);
            border: 1px solid var(--blue);
            box-shadow: 0 20px 48px var(--glow-blue), var(--shadow);
        }}
        .result-card-top {{
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            margin-bottom: 1rem;
        }}
        .result-sentiment-emoji {{ font-size: 2.8rem; line-height: 1; }}
        .result-label {{ font-size: 0.72rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; color: var(--muted); margin-bottom: 0.25rem; }}
        .result-sentiment {{
            font-family: 'Space Grotesk', sans-serif;
            font-size: 2.6rem;
            font-weight: 800;
            line-height: 1.05;
            letter-spacing: -0.02em;
        }}
        .result-sentiment.positive {{ color: var(--green); text-shadow: 0 0 24px var(--glow-green); }}
        .result-sentiment.negative {{ color: var(--red);   text-shadow: 0 0 24px var(--glow-red);   }}
        .result-sentiment.neutral  {{ color: var(--blue);  text-shadow: 0 0 24px var(--glow-blue);  }}
        .confidence-ring-wrap {{
            text-align: center;
        }}
        .confidence-pct {{
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.5rem;
            font-weight: 800;
            color: var(--text);
        }}
        .confidence-sub {{
            font-size: 0.7rem;
            color: var(--muted);
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }}
        .result-insight {{
            font-size: 0.9rem;
            color: var(--muted);
            line-height: 1.7;
            padding-top: 0.75rem;
            border-top: 1px solid var(--border);
        }}
        .result-insight strong {{ color: var(--text); }}

        /* ── Probability Bars ── */
        @keyframes bar-grow {{
            from {{ width: 0%; }}
            to {{ width: var(--bar-w); }}
        }}
        .prob-row {{ margin: 0.8rem 0; }}
        .prob-label {{ display:flex; justify-content:space-between; align-items:center; color:var(--text); font-size:0.88rem; font-weight:600; margin-bottom:0.4rem; }}
        .prob-pct {{ font-family:'Space Grotesk',sans-serif; font-size:0.9rem; font-weight:700; }}
        .prob-track {{ height: 10px; border-radius: 999px; background: var(--panel-soft); border: 1px solid var(--border); overflow: visible; position: relative; }}
        .prob-fill {{
            height: 100%;
            border-radius: 999px;
            animation: bar-grow 0.85s cubic-bezier(0.34, 1.56, 0.64, 1) forwards;
            position: relative;
        }}
        .prob-fill.positive {{ background: linear-gradient(90deg, var(--green), rgba(52,211,153,0.6)); box-shadow: 0 0 12px var(--glow-green); }}
        .prob-fill.negative {{ background: linear-gradient(90deg, var(--red), rgba(248,113,113,0.6));   box-shadow: 0 0 12px var(--glow-red);   }}
        .prob-fill.neutral  {{ background: linear-gradient(90deg, var(--blue), rgba(96,165,250,0.6));   box-shadow: 0 0 12px var(--glow-blue);  }}
        .prob-fill.top      {{ box-shadow: 0 0 18px currentColor; }}

        /* ── Keyword Highlights ── */
        .keyword-card {{
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 1.25rem;
            line-height: 1.9;
            font-size: 0.95rem;
        }}
        .highlight-pos {{ background: rgba(52,211,153,.18); color: var(--green); padding: 1px 7px; border-radius: 6px; font-weight: 700; }}
        .highlight-neg {{ background: rgba(248,113,113,.18); color: var(--red);   padding: 1px 7px; border-radius: 6px; font-weight: 700; }}
        .highlight-neutral {{ color: var(--muted); }}

        /* ── Badges ── */
        .badge {{ display:inline-block; padding:.2rem .6rem; border-radius:999px; font-weight:700; font-size:.8rem; }}
        .badge-positive {{ color: var(--green); background: rgba(52,211,153,.12); border: 1px solid rgba(52,211,153,.30); }}
        .badge-neutral  {{ color: var(--blue);  background: rgba(96,165,250,.12);  border: 1px solid rgba(96,165,250,.30);  }}
        .badge-negative {{ color: var(--red);   background: rgba(248,113,113,.12); border: 1px solid rgba(248,113,113,.30); }}

        /* ── Misc UI ── */
        .small-label {{ color: var(--muted); font-size: .82rem; }}
        .big-value {{ font-family: 'Space Grotesk', sans-serif; font-size: 1.7rem; font-weight: 800; margin-top: .25rem; color: var(--text); }}
        .insight-text {{ color: var(--muted); line-height: 1.7; font-size: 0.9rem; }}

        /* ── Main Buttons (44px Minimum Touch Target) ── */
        .stButton > button, .stDownloadButton > button {{
            background: linear-gradient(135deg, var(--accent), var(--accent-2)) !important;
            color: #FFFFFF !important;
            border: 1px solid rgba(255, 255, 255, 0.25) !important;
            border-radius: 12px !important;
            font-weight: 700 !important;
            font-size: 0.95rem !important;
            font-family: 'Inter', sans-serif !important;
            padding: 0.7rem 1.4rem !important;
            min-height: 44px !important;
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
            letter-spacing: 0.015em !important;
            transition: all 0.22s cubic-bezier(0.4, 0, 0.2, 1) !important;
            box-shadow: 0 4px 16px var(--glow-accent) !important;
            text-shadow: 0 1px 2px rgba(0, 0, 0, 0.15) !important;
        }}

        .stButton > button:focus-visible, .stDownloadButton > button:focus-visible {{
            outline: 2px solid var(--accent) !important;
            outline-offset: 2px !important;
        }}

        .stButton > button *, .stDownloadButton > button * {{
            color: #FFFFFF !important;
            fill: #FFFFFF !important;
            font-weight: 700 !important;
            font-size: 0.95rem !important;
        }}

        .stButton > button:hover, .stDownloadButton > button:hover {{
            filter: brightness(1.15) !important;
            transform: translateY(-2px) !important;
            box-shadow: 0 8px 24px var(--glow-accent) !important;
            border-color: rgba(255, 255, 255, 0.45) !important;
        }}

        .stButton > button:hover *, .stDownloadButton > button:hover * {{
            color: #FFFFFF !important;
        }}

        .stButton > button:active, .stDownloadButton > button:active {{
            transform: translateY(0) !important;
        }}

        /* ── Sidebar Buttons Override (Default / Inactive - 44px Touch Target) ── */
        [data-testid="stSidebar"] .stButton > button,
        [data-testid="stSidebar"] button[kind="secondary"] {{
            background: var(--panel-soft) !important;
            color: var(--text) !important;
            border: 1px solid var(--border) !important;
            box-shadow: none !important;
            text-shadow: none !important;
            font-weight: 600 !important;
            text-align: left !important;
            justify-content: flex-start !important;
            min-height: 44px !important;
            transition: all 0.2s ease !important;
        }}

        [data-testid="stSidebar"] .stButton > button *,
        [data-testid="stSidebar"] button[kind="secondary"] * {{
            color: var(--text) !important;
            fill: var(--text) !important;
            font-weight: 600 !important;
            font-size: 0.88rem !important;
        }}

        [data-testid="stSidebar"] .stButton > button:hover,
        [data-testid="stSidebar"] button[kind="secondary"]:hover {{
            background: linear-gradient(135deg, rgba(99,102,241,0.15), rgba(168,85,247,0.15)) !important;
            border-color: var(--accent) !important;
        }}

        [data-testid="stSidebar"] .stButton > button:hover *,
        [data-testid="stSidebar"] button[kind="secondary"]:hover * {{
            color: var(--accent) !important;
            fill: var(--accent) !important;
        }}

        /* ── Sidebar Active Selected Button (Primary) ── */
        [data-testid="stSidebar"] button[kind="primary"],
        [data-testid="stSidebar"] button[data-testid="stBaseButton-primary"] {{
            background: linear-gradient(135deg, var(--accent), var(--accent-2)) !important;
            color: #FFFFFF !important;
            border: 1px solid rgba(255, 255, 255, 0.35) !important;
            box-shadow: 0 4px 16px var(--glow-accent) !important;
            text-shadow: 0 1px 2px rgba(0,0,0,0.2) !important;
            font-weight: 700 !important;
            text-align: left !important;
            justify-content: flex-start !important;
        }}

        [data-testid="stSidebar"] button[kind="primary"] *,
        [data-testid="stSidebar"] button[data-testid="stBaseButton-primary"] * {{
            color: #FFFFFF !important;
            fill: #FFFFFF !important;
            font-weight: 700 !important;
            font-size: 0.9rem !important;
        }}

        /* ── Inputs ── */
        textarea, input, [data-baseweb="select"] > div {{
            background: var(--input-bg) !important;
            color: var(--text) !important;
            border-color: var(--border) !important;
            border-radius: 10px !important;
            font-family: 'Inter', sans-serif !important;
        }}
        textarea:focus, input:focus {{
            border-color: var(--accent) !important;
            box-shadow: 0 0 0 3px var(--glow-accent) !important;
        }}
        textarea::placeholder, input::placeholder {{ color: var(--muted) !important; opacity: 1 !important; }}
        textarea {{ caret-color: var(--accent) !important; }}

        /* ── Dataframe ── */
        [data-testid="stDataFrame"] {{ border: 1px solid var(--border); border-radius: 14px; overflow: hidden; }}

        /* ── Streamlit Metric (override for fallback) ── */
        [data-testid="stMetric"] {{
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: .85rem;
            box-shadow: var(--shadow);
        }}

        /* ── Alerts ── */
        [data-testid="stAlert"] {{
            border-radius: 12px;
            border: 1px solid var(--border);
            font-family: 'Inter', sans-serif;
        }}

        /* ── File Uploader ── */
        [data-testid="stFileUploader"] {{
            background: transparent !important;
            color: var(--text) !important;
        }}
        [data-testid="stFileUploader"] label,
        [data-testid="stFileUploader"] label p,
        [data-testid="stFileUploader"] p {{ color: var(--text) !important; }}
        [data-testid="stFileUploaderDropzone"],
        [data-testid="stFileUploadDropzone"] {{
            background: var(--panel) !important;
            border: 2px dashed var(--border) !important;
            border-radius: 16px !important;
            color: var(--text) !important;
            transition: border-color 0.2s !important;
        }}
        [data-testid="stFileUploaderDropzone"]:hover,
        [data-testid="stFileUploadDropzone"]:hover {{ border-color: var(--accent) !important; }}
        [data-testid="stFileUploaderDropzone"] p,
        [data-testid="stFileUploaderDropzone"] span,
        [data-testid="stFileUploaderDropzone"] small,
        [data-testid="stFileUploadDropzone"] p,
        [data-testid="stFileUploadDropzone"] span,
        [data-testid="stFileUploadDropzone"] small {{ color: var(--text) !important; }}
        [data-testid="stFileUploaderDropzone"] small,
        [data-testid="stFileUploadDropzone"] small {{ color: var(--muted) !important; }}
        [data-testid="stFileUploaderDropzone"] svg,
        [data-testid="stFileUploadDropzone"] svg {{ color: var(--accent) !important; fill: var(--accent) !important; }}
        [data-testid="stFileUploaderDropzone"] button,
        [data-testid="stFileUploadDropzone"] button {{
            background: var(--panel-soft) !important;
            color: var(--accent) !important;
            border: 1px solid var(--accent) !important;
            border-radius: 10px !important;
            font-weight: 700 !important;
        }}
        [data-testid="stFileUploaderFile"] {{
            background: var(--panel-soft) !important;
            border: 1px solid var(--border) !important;
            border-radius: 12px !important;
            color: var(--text) !important;
        }}
        [data-testid="stFileUploaderFile"] div,
        [data-testid="stFileUploaderFile"] span,
        [data-testid="stFileUploaderFile"] p,
        [data-testid="stFileUploaderFile"] small {{ color: var(--text) !important; }}
        [data-testid="stFileUploaderFile"] small {{ color: var(--muted) !important; }}
        [data-testid="stFileUploaderFile"] svg {{ color: var(--accent) !important; fill: var(--accent) !important; }}
        [data-testid="stFileUploaderFile"] button,
        [data-testid="stFileUploaderDeleteBtn"] {{
            background: transparent !important;
            color: var(--accent) !important;
            border: 1px solid var(--accent) !important;
            border-radius: 999px !important;
        }}

        /* ── Sidebar Nav ── */
        .sidebar-nav-item {{
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 0.55rem 0.75rem;
            border-radius: 10px;
            font-weight: 600;
            font-size: 0.88rem;
            color: var(--muted);
            cursor: pointer;
            transition: background 0.15s, color 0.15s;
            border: none;
            background: transparent;
            width: 100%;
            text-align: left;
        }}
        .sidebar-nav-item:hover {{
            background: var(--panel-soft);
            color: var(--text);
        }}
        .sidebar-nav-item.active {{
            background: linear-gradient(135deg, rgba(129,140,248,0.15), rgba(167,139,250,0.10));
            color: var(--accent);
            border: 1px solid rgba(129,140,248,0.25);
        }}
        .sidebar-info-box {{
            background: var(--panel-soft);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 0.75rem 1rem;
            font-size: 0.78rem;
            color: var(--muted);
            line-height: 1.6;
        }}

        /* ── Timeline (About) ── */
        .timeline {{
            display: flex;
            gap: 0;
            position: relative;
            margin: 1rem 0;
        }}
        .timeline-step {{
            flex: 1;
            position: relative;
            text-align: center;
            padding: 0 0.5rem;
        }}
        .timeline-step::before {{
            content: '';
            position: absolute;
            top: 22px;
            left: calc(50% + 22px);
            right: calc(-50% + 22px);
            height: 2px;
            background: linear-gradient(to right, var(--accent), var(--border));
        }}
        .timeline-step:last-child::before {{ display: none; }}
        .timeline-circle {{
            width: 44px;
            height: 44px;
            border-radius: 50%;
            background: linear-gradient(135deg, var(--accent), var(--accent-2));
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.1rem;
            margin: 0 auto 0.6rem;
            box-shadow: 0 4px 16px var(--glow-accent);
        }}
        .timeline-title {{
            font-family: 'Space Grotesk', sans-serif;
            font-size: 0.82rem;
            font-weight: 700;
            color: var(--text);
            margin-bottom: 0.2rem;
        }}
        .timeline-desc {{
            font-size: 0.72rem;
            color: var(--muted);
            line-height: 1.5;
        }}

        /* ── Caption/Footnote ── */
        .footnote {{ font-size: 0.75rem; color: var(--muted); margin-top: 0.4rem; }}

        /* ── Scrollbar ── */
        ::-webkit-scrollbar {{ width: 6px; height: 6px; }}
        ::-webkit-scrollbar-track {{ background: var(--bg); }}
        ::-webkit-scrollbar-thumb {{ background: var(--border); border-radius: 999px; }}
        ::-webkit-scrollbar-thumb:hover {{ background: var(--accent); }}
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


def sentiment_emoji(sentiment):
    label = safe_text(sentiment).lower()
    if "positive" in label:
        return "✅"
    if "negative" in label:
        return "❌"
    return "➖"


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
    max_key = max(probabilities, key=probabilities.get) if probabilities else None
    rows = [
        ("Negative", probabilities.get("negative", 0.0), "negative"),
        ("Neutral",  probabilities.get("neutral",  0.0), "neutral"),
        ("Positive", probabilities.get("positive", 0.0), "positive"),
    ]
    html_rows = []
    for label, value, cls in rows:
        width = max(0, min(float(value) * 100, 100))
        top_cls = " top" if cls == max_key else ""
        pct_text = probability_percent(value)
        html_rows.append(
            f"""
            <div class="prob-row">
                <div class="prob-label">
                    <span>{html.escape(label)}</span>
                    <span class="prob-pct">{pct_text}</span>
                </div>
                <div class="prob-track">
                    <div class="prob-fill {cls}{top_cls}" style="--bar-w:{width:.2f}%; width:{width:.2f}%;"></div>
                </div>
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


def render_kpi_card(icon, label, value, subtitle="", color="accent"):
    st.markdown(
        f"""
        <div class="kpi-card {color}">
            <div class="kpi-icon">{icon}</div>
            <div class="kpi-label">{html.escape(label)}</div>
            <div class="kpi-value">{html.escape(str(value))}</div>
            <div class="kpi-sub">{html.escape(subtitle)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_model_info_card():
    st.markdown(
        f"""
        <div class="model-card">
            <div class="model-card-icon">🤖</div>
            <div>
                <div class="model-card-label">AI Sentiment Engine</div>
                <div class="model-card-title">Fine-tuned Twitter-RoBERTa</div>
                <div class="model-card-sub">Optimized for customer feedback · Social-style text · Reviews</div>
            </div>
            <div class="model-score-pill">✦ 85.58% F1</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def detect_text_columns(df):
    detected = [col for col in df.columns if str(col).lower().strip() in TEXT_COLUMN_PRIORITY]
    text_like = [col for col in df.columns if df[col].dtype == "object" and col not in detected]
    return detected + text_like


def non_empty_text_rows(df, text_column):
    clean_df = df.dropna(subset=[text_column]).copy()
    clean_df[text_column] = clean_df[text_column].astype(str).str.strip()
    return clean_df[clean_df[text_column] != ""]


def dataframe_to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")


def explain_keywords(text):
    tokens = re.findall(r"\b[a-zA-Z']+\b", safe_text(text).lower())
    positive = [word for word in tokens if word in POSITIVE_WORDS]
    negative = [word for word in tokens if word in NEGATIVE_WORDS]
    neutral  = [word for word in tokens if word not in POSITIVE_WORDS and word not in NEGATIVE_WORDS]

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

    for position, (_, row) in enumerate(df.iterrows(), start=1):
        text = safe_text(row[text_column])
        status.info(f"Analyzing feedback record {position} of {total}…")
        prediction = run_ai_prediction(text)
        probabilities = prediction["probabilities"]
        highest_cat = highest_probability_category(probabilities)
        rows.append(
            {
                "original_text": text,
                "final_sentiment": prediction["sentiment"],
                "confidence": prediction["confidence"],
                "negative_probability": probabilities["negative"],
                "neutral_probability":  probabilities["neutral"],
                "positive_probability": probabilities["positive"],
                "highest_probability_category": highest_cat,
            }
        )
        progress.progress(position / total)

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
            font=dict(family="Inter, sans-serif", color=palette["text"], size=11),
            title=dict(font=dict(family="Space Grotesk, sans-serif", color=palette["text"], size=14)),
            legend=dict(font=dict(color=palette["text"], size=11)),
            margin=dict(l=20, r=20, t=55, b=20),
        )
    else:
        fig.update_layout(
            paper_bgcolor="#07111f",
            plot_bgcolor="#0f1b2d",
            font=dict(family="Inter, sans-serif", color="#f0f6ff", size=11),
            title=dict(font=dict(family="Space Grotesk, sans-serif", color="#f0f6ff", size=14)),
            legend=dict(font=dict(color="#f0f6ff", size=11)),
            margin=dict(l=20, r=20, t=55, b=20),
        )
    fig.update_xaxes(
        gridcolor=palette["grid"],
        color=palette["text"],
        tickfont=dict(color=palette["text"]),
        title_font=dict(color=palette["text"]),
    )
    fig.update_yaxes(
        gridcolor=palette["grid"],
        color=palette["text"],
        tickfont=dict(color=palette["text"]),
        title_font=dict(color=palette["text"]),
    )
    return fig


def sidebar():
    pages = ["Overview", "Analyze Text", "Batch Analysis", "Insights Dashboard", "About"]
    if "page" not in st.session_state:
        st.session_state.page = "Overview"
    if "next_page" in st.session_state:
        st.session_state.page = st.session_state.pop("next_page")

    # ── Brand ──
    st.sidebar.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-logo-ring">💬</div>
            <div>
                <div class="sidebar-brand-name">SentiScope AI</div>
                <div class="sidebar-brand-sub">Sentiment Analytics Studio</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Engine status badge ──
    st.sidebar.markdown(
        '<div class="engine-badge"><div class="engine-badge-dot"></div>Engine Ready</div>',
        unsafe_allow_html=True,
    )

    # ── Theme toggle ──
    selected_theme = st.sidebar.radio(
        "🎨 Theme",
        ["Dark Mode", "Light Mode"],
        index=0 if st.session_state.theme == "Dark Mode" else 1,
    )
    if selected_theme != st.session_state.theme:
        st.session_state.theme = selected_theme
        st.rerun()

    st.sidebar.markdown("---")

    # ── Navigation ──
    st.sidebar.markdown("**Navigate**")
    for page_name in pages:
        icon = PAGE_ICONS[page_name]
        is_active = st.session_state.page == page_name
        active_cls = "active" if is_active else ""
        # Use actual Streamlit button per page for interactivity
        if st.sidebar.button(
            f"{icon}  {page_name}",
            key=f"nav_{page_name}",
            use_container_width=True,
            type="primary" if is_active else "secondary",
        ):
            if st.session_state.page != page_name:
                st.session_state.page = page_name
                st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        <div class="sidebar-info-box">
            🧠 Analyzes reviews, comments, support tickets, and survey feedback with a fine-tuned transformer sentiment engine.
        </div>
        """,
        unsafe_allow_html=True,
    )

    return st.session_state.page


def overview_page():
    # ── Animated Hero ──
    st.markdown(
        """
        <div class="hero">
            <div class="hero-eyebrow">✦ AI-Powered Analytics</div>
            <div class="hero-title">
                SentiScope <span class="accent-word">AI Studio</span>
            </div>
            <div class="subtitle">
                Transform unstructured customer feedback into actionable sentiment insights — instantly. Powered by a fine-tuned transformer model.
            </div>
            <div class="hero-stats">
                <div class="hero-stat"><span class="hero-stat-icon">🎯</span><strong>3</strong>&nbsp;Sentiment Classes</div>
                <div class="hero-stat"><span class="hero-stat-icon">⚡</span><strong>85.58%</strong>&nbsp;Balanced F1</div>
                <div class="hero-stat"><span class="hero-stat-icon">🔄</span><strong>Single &amp; Batch</strong>&nbsp;Analysis</div>
                <div class="hero-stat"><span class="hero-stat-icon">📤</span><strong>CSV</strong>&nbsp;Export Ready</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Feature Navigation Cards ──
    st.markdown('<div class="section-title">🚀 Get Started</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            """
            <div class="nav-card">
                <span class="nav-card-icon">🔍</span>
                <div class="nav-card-label">Single Analysis</div>
                <div class="nav-card-title">Analyze Text</div>
                <div class="nav-card-desc">Paste any review, tweet, or support message and get instant sentiment with confidence score and keyword explanation.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("→ Go to Analyze Text", key="cta_analyze", use_container_width=True):
            st.session_state.page = "Analyze Text"
            st.rerun()

    with c2:
        st.markdown(
            """
            <div class="nav-card">
                <span class="nav-card-icon">📊</span>
                <div class="nav-card-label">Batch Processing</div>
                <div class="nav-card-title">Batch Analysis</div>
                <div class="nav-card-desc">Upload a CSV file of customer feedback and generate sentiment predictions across the entire dataset at once.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("→ Go to Batch Analysis", key="cta_batch", use_container_width=True):
            st.session_state.page = "Batch Analysis"
            st.rerun()

    with c3:
        st.markdown(
            """
            <div class="nav-card">
                <span class="nav-card-icon">📈</span>
                <div class="nav-card-label">Insights & Trends</div>
                <div class="nav-card-title">Insights Dashboard</div>
                <div class="nav-card-desc">Explore sentiment trends, KPI metrics, distribution charts, and confidence histograms from your analyzed feedback.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("→ View Dashboard", key="cta_dashboard", use_container_width=True):
            st.session_state.page = "Insights Dashboard"
            st.rerun()

    st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)
    render_model_info_card()

    # ── What it does ──
    st.markdown('<div class="section-title">💡 What This Product Does</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="card">
            <p class="insight-text">
                SentiScope AI turns unstructured customer feedback into sentiment insights. It supports fast single-text analysis,
                CSV-based batch processing, keyword-level explanations, and downloadable results for client reporting or internal
                decision-making. The AI engine is optimized for real-world customer language — not just formal reviews.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def single_analysis_page():
    st.markdown('<div class="section-title">🔍 Analyze Text</div>', unsafe_allow_html=True)
    st.markdown('<p class="insight-text">Analyze a single customer review, comment, tweet, or support message.</p>', unsafe_allow_html=True)
    render_model_info_card()

    st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)

    # ── Sample buttons ──
    st.markdown("**Try a sample:**")
    cols = st.columns(4)
    for idx, (label, text) in enumerate(SAMPLE_TEXTS.items()):
        with cols[idx]:
            if st.button(label, key=f"sample_{idx}"):
                st.session_state.single_text = text

    user_text = st.text_area(
        "Customer feedback",
        key="single_text",
        height=150,
        placeholder="Enter a product review, app comment, survey response, or support message...",
    )

    if st.button("⚡ Analyze Sentiment", use_container_width=True):
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
        emoji = sentiment_emoji(sentiment)
        conf_pct = f"{float(prediction['confidence']) * 100:.1f}%"
        proc_ms = f"{prediction['processing_time'] * 1000:.0f}ms"

        # ── Premium result card ──
        st.markdown(
            f"""
            <div class="result-card {card_class}">
                <div class="result-card-top">
                    <div>
                        <div class="result-label">Final Sentiment</div>
                        <div class="result-sentiment {card_class}">{emoji} {html.escape(sentiment)}</div>
                    </div>
                    <div class="confidence-ring-wrap">
                        <div class="confidence-pct">{conf_pct}</div>
                        <div class="confidence-sub">Confidence</div>
                    </div>
                </div>
                <div class="result-insight">
                    <strong>Insight:</strong> {sentiment_insight(sentiment)}
                    <br><span class="footnote">⏱ Processed in {proc_ms} · Model: {html.escape(safe_text(prediction.get("engine", ""))[:60])}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Probability bars ──
        st.markdown('<div class="section-title">📊 Probability Breakdown</div>', unsafe_allow_html=True)
        render_probability_bars(prediction["probabilities"])

        # ── Keyword explanation ──
        st.markdown('<div class="section-title">🔤 Keyword-Level Explanation</div>', unsafe_allow_html=True)
        highlighted, positive, negative, neutral = explain_keywords(user_text)
        st.markdown(f'<div class="keyword-card">{highlighted}</div>', unsafe_allow_html=True)

        k1, k2, k3 = st.columns(3)
        k1.success("✅ Positive: " + (", ".join(sorted(set(positive))) if positive else "None found"))
        k2.error("❌ Negative: " + (", ".join(sorted(set(negative))) if negative else "None found"))
        k3.info(f"➖ Neutral words: {len(neutral)}")
        st.markdown(
            '<p class="footnote">Keywords provide context clues. The final prediction is generated by the AI sentiment engine, not keyword matching alone.</p>',
            unsafe_allow_html=True,
        )


def batch_page():
    st.markdown('<div class="section-title">📊 Batch Analysis</div>', unsafe_allow_html=True)
    st.markdown('<p class="insight-text">Upload CSV feedback to analyze sentiment at scale. Supports Xquik exports with tweet_text or full_text columns.</p>', unsafe_allow_html=True)
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
        st.error("No feedback text column found. Please upload a CSV with columns like: text, review, tweet_text, full_text, comment, feedback, content, or message.")
        return

    st.markdown("**Select the column that contains the review, comment, tweet, or feedback text you want to analyze.**")

    if len(text_columns) == 1:
        selected_column = text_columns[0]
        st.success(f"✓ Auto-detected text column: **{selected_column}**")
    else:
        selected_column = st.selectbox("Text column", text_columns, help="Choose the column with customer feedback text.")

    st.markdown("**Preview of selected text column:**")
    sample_texts = df[selected_column].dropna().head(3).values
    for i, sample in enumerate(sample_texts, 1):
        st.caption(f"Sample {i}: {safe_text(sample)[:100]}…" if len(safe_text(sample)) > 100 else f"Sample {i}: {sample}")

    blank_text_count = int(df[selected_column].fillna("").astype(str).str.strip().eq("").sum())

    # ── KPI metrics ──
    c1, c2, c3 = st.columns(3)
    with c1:
        render_kpi_card("📄", "Total Rows", len(df), "records in file", "accent")
    with c2:
        render_kpi_card("⬜", "Blank Text", blank_text_count, "will be skipped", "blue")
    with c3:
        render_kpi_card("🗂️", "Columns", len(df.columns), "in dataset", "accent")

    st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)
    st.markdown("**Dataset Preview**")
    st.dataframe(df.head(8), use_container_width=True)

    if st.button("⚡ Run Feedback Sentiment Analysis", use_container_width=True):
        clean_df = non_empty_text_rows(df, selected_column)
        if clean_df.empty:
            st.warning("No feedback text rows found after removing blank values.")
            return
        with st.spinner("Analyzing feedback with AI. This may take a moment."):
            results = run_batch(clean_df, selected_column)
        st.session_state.batch_results = results
        st.success("✓ Analysis completed successfully.")

    results = st.session_state.get("batch_results")
    if results is not None and not results.empty:
        st.markdown('<div class="section-title">📋 Prediction Results</div>', unsafe_allow_html=True)

        display_results = results.copy()
        display_results.columns = [
            "Text", "Sentiment", "Confidence", "Negative %", "Neutral %", "Positive %", "Category"
        ]
        st.dataframe(display_results, use_container_width=True)

        export_df = clean_export_dataframe(results)
        st.download_button(
            "⬇ Download Results CSV",
            data=dataframe_to_csv_bytes(export_df),
            file_name="sentiscope_analysis_results.csv",
            mime="text/csv",
        )
        if st.button("📈 View Insights Dashboard", use_container_width=True):
            st.session_state.next_page = "Insights Dashboard"
            st.session_state.scroll_to_top = True
            st.rerun()


def analytics_page():
    if st.session_state.get("scroll_to_top"):
        scroll_to_top()
        st.session_state.scroll_to_top = False

    st.markdown('<div class="section-title">📈 Customer Sentiment Overview</div>', unsafe_allow_html=True)
    render_model_info_card()

    results = st.session_state.get("batch_results")
    if results is None or results.empty:
        st.info("No analyzed feedback results found yet. Upload a CSV in Batch Analysis and run sentiment analysis to see dashboard insights.")
        return

    sentiment_counts = results["final_sentiment"].value_counts()
    avg_conf = results["confidence"].mean()
    most_common = sentiment_counts.idxmax() if not sentiment_counts.empty else "N/A"

    total_feedback   = len(results)
    positive_count   = int(sentiment_counts.get("Positive", 0))
    neutral_count    = int(sentiment_counts.get("Neutral", 0))
    negative_count   = int(sentiment_counts.get("Negative", 0))

    # ── KPI Cards row 1 ──
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        render_kpi_card("📋", "Total Analyzed", total_feedback, "feedback records", "accent")
    with k2:
        render_kpi_card("✅", "Positive", positive_count, f"{positive_count/total_feedback*100:.0f}% of total" if total_feedback else "", "green")
    with k3:
        render_kpi_card("➖", "Neutral", neutral_count, f"{neutral_count/total_feedback*100:.0f}% of total" if total_feedback else "", "blue")
    with k4:
        render_kpi_card("❌", "Negative", negative_count, f"{negative_count/total_feedback*100:.0f}% of total" if total_feedback else "", "red")

    st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)

    # ── KPI Cards row 2 ──
    k5, k6 = st.columns(2)
    with k5:
        render_kpi_card("🎯", "Avg Confidence", format_confidence(avg_conf), "across all predictions", "accent")
    with k6:
        dominant_emoji = sentiment_emoji(most_common)
        render_kpi_card(dominant_emoji, "Dominant Sentiment", most_common, "most frequent class", "accent")

    theme = st.session_state.get("theme", "Dark Mode")
    colors = {"Positive": "#34d399", "Negative": "#f87171", "Neutral": "#60a5fa"}
    chart_df = sentiment_counts.reset_index()
    chart_df.columns = ["Sentiment", "Count"]

    # ── Charts ──
    st.markdown('<div class="section-title">🥧 Feedback Mood Breakdown</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        fig = px.pie(
            chart_df, names="Sentiment", values="Count",
            title="Sentiment Distribution",
            color="Sentiment", color_discrete_map=colors,
            hole=0.45,
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(style_chart(fig, theme), use_container_width=True)
    with c2:
        fig = px.bar(
            chart_df, x="Sentiment", y="Count",
            title="Sentiment Counts",
            color="Sentiment", color_discrete_map=colors, text="Count",
        )
        fig.update_traces(marker_line_width=0, textposition="outside")
        st.plotly_chart(style_chart(fig, theme), use_container_width=True)

    st.markdown('<div class="section-title">📉 Confidence Distribution</div>', unsafe_allow_html=True)
    fig = px.histogram(
        results, x="confidence", nbins=12,
        title="Model Confidence Scores",
        color="final_sentiment", color_discrete_map=colors,
    )
    fig.update_xaxes(tickformat=".0%", title="Confidence")
    st.plotly_chart(style_chart(fig, theme), use_container_width=True)

    st.markdown('<div class="section-title">🔎 Key Sentiment Insights</div>', unsafe_allow_html=True)
    st.dataframe(results.head(25), use_container_width=True)


def about_page():
    st.markdown('<div class="section-title">ℹ️ About SentiScope AI Studio</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="card">
            <p class="insight-text">
                SentiScope AI Studio is a client-ready sentiment analytics dashboard for customer feedback. It helps teams understand how users
                feel about products, services, and support experiences. Upload reviews, comments, or survey responses, and get instant sentiment
                insights with exportable results for business reporting.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)
    render_model_info_card()

    st.markdown('<div class="section-title">⚙️ Technical Details</div>', unsafe_allow_html=True)
    t1, t2 = st.columns(2)
    with t1:
        st.markdown(
            """
            <div class="card">
                <div class="small-label">Model Architecture</div>
                <div class="big-value" style="font-size:1.1rem;">Twitter-RoBERTa</div>
                <p class="insight-text">Fine-tuned transformer model, optimized for customer feedback, reviews, and social-style text. Falls back to <code>cardiffnlp/twitter-roberta-base-sentiment-latest</code> if local model is unavailable.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with t2:
        st.markdown(
            """
            <div class="card">
                <div class="small-label">Balanced Quality Score (Macro F1)</div>
                <div class="big-value" style="color:var(--green);">85.58%</div>
                <p class="insight-text">Measures consistent performance across Positive, Neutral, and Negative classes equally. Evaluated on a held-out test set.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="card" style="margin-top:0.75rem;">
            <div class="small-label" style="margin-bottom:0.5rem;">Performance Metrics</div>
            <table style="width:100%;border-collapse:collapse;font-size:0.88rem;">
                <thead>
                    <tr style="border-bottom:1px solid var(--border);">
                        <th style="text-align:left;padding:0.5rem 0.75rem;color:var(--muted);font-weight:600;">Metric</th>
                        <th style="text-align:right;padding:0.5rem 0.75rem;color:var(--muted);font-weight:600;">Score</th>
                    </tr>
                </thead>
                <tbody>
                    <tr><td style="padding:0.45rem 0.75rem;">Accuracy</td><td style="text-align:right;padding:0.45rem 0.75rem;font-weight:700;">85.53%</td></tr>
                    <tr style="background:var(--panel-soft);"><td style="padding:0.45rem 0.75rem;border-radius:6px 0 0 6px;">Macro Precision</td><td style="text-align:right;padding:0.45rem 0.75rem;border-radius:0 6px 6px 0;font-weight:700;">85.88%</td></tr>
                    <tr><td style="padding:0.45rem 0.75rem;">Macro Recall</td><td style="text-align:right;padding:0.45rem 0.75rem;font-weight:700;">85.53%</td></tr>
                    <tr style="background:var(--panel-soft);"><td style="padding:0.45rem 0.75rem;border-radius:6px 0 0 6px;color:var(--green);font-weight:700;">Balanced Quality (F1)</td><td style="text-align:right;padding:0.45rem 0.75rem;border-radius:0 6px 6px 0;color:var(--green);font-weight:800;">85.58%</td></tr>
                </tbody>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Workflow Timeline ──
    st.markdown('<div class="section-title">🗺️ Recommended Workflow</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="timeline">
            <div class="timeline-step">
                <div class="timeline-circle">🔍</div>
                <div class="timeline-title">Test Single Examples</div>
                <div class="timeline-desc">Use Analyze Text with real feedback to understand model behavior</div>
            </div>
            <div class="timeline-step">
                <div class="timeline-circle">📊</div>
                <div class="timeline-title">Upload CSV Feedback</div>
                <div class="timeline-desc">Batch analyze reviews, comments, or support tickets at scale</div>
            </div>
            <div class="timeline-step">
                <div class="timeline-circle">📈</div>
                <div class="timeline-title">Review Dashboard</div>
                <div class="timeline-desc">Explore KPIs, sentiment charts, and confidence distributions</div>
            </div>
            <div class="timeline-step">
                <div class="timeline-circle">📤</div>
                <div class="timeline-title">Export Results</div>
                <div class="timeline-desc">Download clean CSV for client reports or business reviews</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


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
