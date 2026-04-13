"""
Professional Government-Grade Streamlit Application
Citizen Grievance Analysis System - Enhanced Dark Theme Edition
Built with enterprise-grade styling, advanced animations, and accessibility standards
"""

from __future__ import annotations

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any
import requests
from datetime import datetime
import time

import streamlit as st


# Configuration
BASE_DIR = Path(__file__).resolve().parent
SENTIMENT_MODEL_PATH = BASE_DIR / "trained_models" / "sentiment_model.joblib"
VECTORIZER_MODEL_PATH = BASE_DIR / "trained_models" / "tfidf_vectorizer.joblib"
METADATA_PATH = BASE_DIR / "trained_models" / "model_metadata.json"
API_BASE_URL = "http://localhost:8000"

# Color Palette - Government Professional Theme
COLORS = {
    "primary": "#0A66C2",           # Professional Blue
    "primary_dark": "#004B87",      # Darker Blue
    "primary_light": "#1F8FE9",     # Light Blue
    "accent": "#FF6B35",             # Orange accent
    "success": "#06A77D",            # Green
    "warning": "#F7A072",            # Warning orange
    "danger": "#D62828",             # Red
    "dark_bg": "#0F1419",            # Very dark background
    "card_bg": "#1A1F26",            # Card background
    "card_hover": "#242B33",         # Card hover color
    "border": "#2D3748",             # Border color
    "text_primary": "#FFFFFF",       # Primary text
    "text_secondary": "#C0C0C0",     # Secondary text
    "text_muted": "#808080",         # Muted text
    "gradient_start": "#0A66C2",     # Gradient start
    "gradient_end": "#004B87",       # Gradient end
}

# Priority Mapping
PRIORITY_MAPPING = {
    "Critical": {
        "score": 5,
        "label": "CRITICAL",
        "description": "Requires immediate action",
        "icon": "⚠️",
        "color": COLORS["danger"]
    },
    "Negative": {
        "score": 4,
        "label": "HIGH",
        "description": "Should be addressed promptly",
        "icon": "⬆️",
        "color": COLORS["warning"]
    },
    "Neutral": {
        "score": 3,
        "label": "MEDIUM",
        "description": "Standard processing",
        "icon": "→",
        "color": COLORS["primary"]
    },
    "Positive": {
        "score": 2,
        "label": "LOW",
        "description": "Routine handling",
        "icon": "✓",
        "color": COLORS["success"]
    }
}


@st.cache_resource
def load_models():
    """Load pre-trained models and metadata"""
    try:
        model = joblib.load(SENTIMENT_MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_MODEL_PATH)
        with open(METADATA_PATH) as f:
            metadata = json.load(f)
        return model, vectorizer, metadata
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None


def predict_sentiment(complaint_text: str, model, vectorizer) -> Dict[str, Any]:
    """Predict sentiment for complaint text"""
    try:
        features = vectorizer.transform([complaint_text])
        prediction = model.predict(features)[0]
        probabilities = model.decision_function(features)[0]
        
        priority_info = PRIORITY_MAPPING.get(prediction, {})
        
        return {
            "sentiment": prediction,
            "priority_score": priority_info.get("score", 0),
            "priority_label": priority_info.get("label", "UNKNOWN"),
            "color": priority_info.get("color", "#666666"),
            "icon": priority_info.get("icon", "?"),
            "description": priority_info.get("description", "Unknown priority")
        }
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return {"sentiment": "Error", "priority_score": 0, "priority_label": "ERROR", "color": "#999999"}


def test_api_connection() -> bool:
    """Test if API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False


# PAGE CONFIGURATION
st.set_page_config(
    page_title="Citizen Grievance Analysis Portal",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# COMPREHENSIVE ENHANCED DARK THEME CSS WITH ANIMATIONS
st.markdown(f"""
<style>
    :root {{
        --primary-color: {COLORS["primary"]};
        --primary-dark: {COLORS["primary_dark"]};
        --primary-light: {COLORS["primary_light"]};
        --accent-color: {COLORS["accent"]};
        --success-color: {COLORS["success"]};
        --warning-color: {COLORS["warning"]};
        --danger-color: {COLORS["danger"]};
        --dark-bg: {COLORS["dark_bg"]};
        --card-bg: {COLORS["card_bg"]};
        --card-hover: {COLORS["card_hover"]};
        --border-color: {COLORS["border"]};
        --text-primary: {COLORS["text_primary"]};
        --text-secondary: {COLORS["text_secondary"]};
        --text-muted: {COLORS["text_muted"]};
    }}

    * {{
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }}

    /* Main theme */
    .main {{
        background-color: {COLORS["dark_bg"]};
        color: {COLORS["text_primary"]};
    }}

    [data-testid="stAppViewContainer"] {{
        background-color: {COLORS["dark_bg"]};
    }}

    [data-testid="stSidebar"] {{
        background-color: {COLORS["card_bg"]};
        border-right: 2px solid {COLORS["border"]};
    }}

    /* Typography */
    body, p, span, div {{
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen', 
                     'Ubuntu', 'Cantarell', 'Fira Sans', 'Droid Sans', sans-serif;
    }}

    h1, h2, h3, h4, h5, h6 {{
        font-weight: 600;
        letter-spacing: -0.5px;
    }}

    /* ANIMATIONS */
    @keyframes fadeIn {{
        from {{
            opacity: 0;
        }}
        to {{
            opacity: 1;
        }}
    }}

    @keyframes slideInDown {{
        from {{
            opacity: 0;
            transform: translateY(-20px);
        }}
        to {{
            opacity: 1;
            transform: translateY(0);
        }}
    }}

    @keyframes slideInUp {{
        from {{
            opacity: 0;
            transform: translateY(20px);
        }}
        to {{
            opacity: 1;
            transform: translateY(0);
        }}
    }}

    @keyframes slideInRight {{
        from {{
            opacity: 0;
            transform: translateX(20px);
        }}
        to {{
            opacity: 1;
            transform: translateX(0);
        }}
    }}

    @keyframes slideInLeft {{
        from {{
            opacity: 0;
            transform: translateX(-20px);
        }}
        to {{
            opacity: 1;
            transform: translateX(0);
        }}
    }}

    @keyframes slideInRight {{
        from {{
            opacity: 0;
            transform: translateX(20px);
        }}
        to {{
            opacity: 1;
            transform: translateX(0);
        }}
    }}

    @keyframes scale {{
        from {{
            transform: scale(0.95);
            opacity: 0;
        }}
        to {{
            transform: scale(1);
            opacity: 1;
        }}
    }}

    @keyframes pulse {{
        0%, 100% {{ opacity: 1; }}
        50% {{ opacity: 0.5; }}
    }}

    @keyframes glow {{
        0%, 100% {{
            box-shadow: 0 0 10px rgba(10, 102, 194, 0.3), 0 4px 12px rgba(0, 0, 0, 0.3);
        }}
        50% {{
            box-shadow: 0 0 20px rgba(10, 102, 194, 0.6), 0 4px 12px rgba(0, 0, 0, 0.3);
        }}
    }}

    @keyframes shimmer {{
        0% {{
            background-position: -1000px 0;
        }}
        100% {{
            background-position: 1000px 0;
        }}
    }}

    @keyframes countUp {{
        from {{
            opacity: 0;
            transform: scale(0.5) rotateZ(-10deg);
        }}
        to {{
            opacity: 1;
            transform: scale(1) rotateZ(0deg);
        }}
    }}

    /* Government Header */
    .gov-header {{
        background: linear-gradient(135deg, {COLORS["primary_dark"]} 0%, {COLORS["primary"]} 100%);
        color: {COLORS["text_primary"]};
        padding: 50px 40px;
        border-radius: 8px;
        margin-bottom: 40px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4), 0 0 1px rgba(10, 102, 194, 0.3);
        border-left: 6px solid {COLORS["accent"]};
        animation: slideInDown 0.6s ease-out;
        position: relative;
        overflow: hidden;
    }}

    .gov-header::before {{
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
        animation: shimmer 3s infinite;
    }}

    .gov-header h1 {{
        font-size: 2.8em;
        margin: 0 0 12px 0;
        font-weight: 800;
        letter-spacing: -1.5px;
        position: relative;
        z-index: 2;
    }}

    .gov-header p {{
        font-size: 1.1em;
        opacity: 0.95;
        margin: 0;
        position: relative;
        z-index: 2;
        font-weight: 300;
    }}

    .gov-seal {{
        display: inline-block;
        font-size: 3em;
        margin-right: 20px;
        vertical-align: middle;
        animation: scale 0.8s ease-out 0.2s both;
    }}

    /* Status Bar */
    .status-bar {{
        background-color: {COLORS["card_bg"]};
        border: 1px solid {COLORS["border"]};
        border-radius: 8px;
        padding: 18px 24px;
        margin-bottom: 30px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        animation: slideInDown 0.6s ease-out 0.1s both;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    }}

    .status-bar:hover {{
        border-color: {COLORS["primary"]};
        box-shadow: 0 6px 16px rgba(10, 102, 194, 0.2);
    }}

    .status-item {{
        display: flex;
        align-items: center;
        gap: 12px;
        animation: slideInLeft 0.6s ease-out;
    }}

    .status-badge {{
        display: inline-block;
        width: 14px;
        height: 14px;
        border-radius: 50%;
        background-color: {COLORS["success"]};
        animation: pulse 2s infinite;
        box-shadow: 0 0 8px rgba(6, 168, 125, 0.6);
    }}

    .status-badge.error {{
        background-color: {COLORS["danger"]};
        box-shadow: 0 0 8px rgba(214, 40, 40, 0.6);
    }}

    /* Input Section */
    .input-container {{
        background-color: {COLORS["card_bg"]};
        border: 2px solid {COLORS["border"]};
        border-radius: 8px;
        padding: 30px;
        margin-bottom: 30px;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        animation: slideInUp 0.6s ease-out 0.2s both;
        position: relative;
    }}

    .input-container::before {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, {COLORS["primary"]}, transparent);
        opacity: 0;
        transition: opacity 0.3s ease;
    }}

    .input-container:hover {{
        border-color: {COLORS["primary"]};
        box-shadow: 0 8px 20px rgba(10, 102, 194, 0.15);
        background-color: {COLORS["card_hover"]};
    }}

    .input-container:focus-within {{
        border-color: {COLORS["primary"]};
        box-shadow: 0 0 0 3px rgba(10, 102, 194, 0.15);
    }}

    .input-label {{
        display: block;
        font-weight: 600;
        margin-bottom: 12px;
        color: {COLORS["text_primary"]};
        font-size: 0.95em;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        animation: slideInLeft 0.5s ease-out;
    }}

    .input-help {{
        font-size: 0.85em;
        color: {COLORS["text_muted"]};
        margin-top: 8px;
        animation: fadeIn 0.5s ease-out 0.2s both;
    }}

    /* Text Area */
    .stTextArea textarea {{
        background-color: {COLORS["dark_bg"]} !important;
        color: {COLORS["text_primary"]} !important;
        border: 1.5px solid {COLORS["border"]} !important;
        border-radius: 6px !important;
        font-family: 'Segoe UI', sans-serif !important;
        font-size: 1em !important;
        padding: 14px !important;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }}

    .stTextArea textarea:focus {{
        border-color: {COLORS["primary"]} !important;
        box-shadow: 0 0 0 3px rgba(10, 102, 194, 0.2) !important;
        background-color: {COLORS["card_bg"]} !important;
    }}

    .stTextArea textarea:hover {{
        border-color: {COLORS["primary_light"]} !important;
    }}

    /* Button */
    .stButton > button {{
        background: linear-gradient(135deg, {COLORS["primary"]} 0%, {COLORS["primary_dark"]} 100%);
        color: {COLORS["text_primary"]};
        border: none;
        border-radius: 6px;
        padding: 16px 40px;
        font-weight: 700;
        font-size: 1em;
        cursor: pointer;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 6px 16px rgba(10, 102, 194, 0.35);
        text-transform: uppercase;
        letter-spacing: 1.2px;
        min-width: 180px;
        position: relative;
        overflow: hidden;
    }}

    .stButton > button::before {{
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 0;
        height: 0;
        background: rgba(255, 255, 255, 0.2);
        border-radius: 50%;
        transform: translate(-50%, -50%);
        transition: width 0.6s, height 0.6s;
    }}

    .stButton > button:hover {{
        transform: translateY(-3px);
        box-shadow: 0 10px 24px rgba(10, 102, 194, 0.5);
    }}

    .stButton > button:hover::before {{
        width: 300px;
        height: 300px;
    }}

    .stButton > button:active {{
        transform: translateY(-1px);
    }}

    /* Results Container */
    .results-box {{
        background: linear-gradient(135deg, {COLORS["card_bg"]} 0%, {COLORS["card_hover"]} 100%);
        border: 1px solid {COLORS["border"]};
        border-radius: 8px;
        padding: 40px;
        margin-top: 40px;
        animation: slideInUp 0.6s ease-out;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
        position: relative;
        overflow: hidden;
    }}

    .results-box::before {{
        content: '';
        position: absolute;
        top: -1px;
        left: 0;
        right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, {COLORS["primary"]}, transparent);
    }}

    /* Sentiment Badge */
    .sentiment-badge {{
        display: inline-block;
        padding: 16px 32px;
        border-radius: 6px;
        font-weight: 800;
        font-size: 1.3em;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 30px;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.4);
        animation: scale 0.5s cubic-bezier(0.34, 1.56, 0.64, 1) both;
        position: relative;
        overflow: hidden;
    }}

    .sentiment-badge::after {{
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent);
        animation: shimmer 2s infinite;
    }}

    /* Metric Card */
    .metric-card {{
        background-color: {COLORS["dark_bg"]};
        border: 1.5px solid {COLORS["border"]};
        border-left: 4px solid {COLORS["primary"]};
        border-radius: 8px;
        padding: 25px;
        text-align: center;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        animation: scale 0.5s cubic-bezier(0.34, 1.56, 0.64, 1) both;
        position: relative;
    }}

    .metric-card:nth-child(1) {{ animation-delay: 0.1s; }}
    .metric-card:nth-child(2) {{ animation-delay: 0.2s; }}
    .metric-card:nth-child(3) {{ animation-delay: 0.3s; }}
    .metric-card:nth-child(4) {{ animation-delay: 0.4s; }}

    .metric-card::before {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(135deg, rgba(10, 102, 194, 0.1) 0%, transparent 100%);
        border-radius: 8px;
        opacity: 0;
        transition: opacity 0.3s ease;
    }}

    .metric-card:hover {{
        border-left-color: {COLORS["accent"]};
        box-shadow: 0 10px 28px rgba(10, 102, 194, 0.3);
        background-color: {COLORS["card_hover"]};
        transform: translateY(-4px);
    }}

    .metric-card:hover::before {{
        opacity: 1;
    }}

    .metric-label {{
        display: block;
        color: {COLORS["text_muted"]};
        font-size: 0.85em;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 12px;
        position: relative;
        z-index: 1;
    }}

    .metric-value {{
        display: block;
        color: {COLORS["primary_light"]};
        font-size: 2.2em;
        font-weight: 800;
        margin-bottom: 8px;
        position: relative;
        z-index: 1;
    }}

    .metric-unit {{
        color: {COLORS["text_muted"]};
        font-size: 0.9em;
        position: relative;
        z-index: 1;
    }}

    /* Section Header */
    .section-title {{
        font-size: 1.6em;
        font-weight: 800;
        color: {COLORS["text_primary"]};
        margin-top: 40px;
        margin-bottom: 25px;
        padding-bottom: 18px;
        border-bottom: 3px solid {COLORS["primary"]};
        text-transform: uppercase;
        letter-spacing: 1.5px;
        animation: slideInLeft 0.6s ease-out;
        position: relative;
    }}

    .section-title::after {{
        content: '';
        position: absolute;
        bottom: -3px;
        left: 0;
        height: 3px;
        background: linear-gradient(90deg, {COLORS["primary"]}, {COLORS["accent"]}, transparent);
        width: 0;
        animation: slideInRight 0.8s ease-out 0.3s forwards;
    }}

    /* Sidebar */
    .sidebar-section {{
        margin-bottom: 28px;
        padding-bottom: 22px;
        border-bottom: 1px solid {COLORS["border"]};
        animation: fadeIn 0.5s ease-out;
    }}

    .sidebar-section:last-child {{
        border-bottom: none;
    }}

    .sidebar-title {{
        font-size: 1.15em;
        font-weight: 800;
        color: {COLORS["primary_light"]};
        margin-bottom: 16px;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        animation: slideInLeft 0.5s ease-out;
    }}

    /* Info Box */
    .info-box {{
        background-color: {COLORS["card_bg"]};
        border-left: 5px solid {COLORS["primary"]};
        border-radius: 6px;
        padding: 18px;
        margin: 16px 0;
        color: {COLORS["text_secondary"]};
        animation: slideInLeft 0.5s ease-out;
        transition: all 0.3s ease;
    }}

    .info-box:hover {{
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
        transform: translateX(2px);
    }}

    .info-box.success {{
        border-left-color: {COLORS["success"]};
    }}

    .info-box.warning {{
        border-left-color: {COLORS["warning"]};
    }}

    .info-box.error {{
        border-left-color: {COLORS["danger"]};
    }}

    /* Table */
    .dataframe {{
        background-color: transparent !important;
        color: {COLORS["text_primary"]} !important;
        border-collapse: collapse !important;
    }}

    .dataframe th {{
        background: linear-gradient(135deg, {COLORS["primary"]} 0%, {COLORS["primary_dark"]} 100%) !important;
        color: {COLORS["text_primary"]} !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.8px !important;
        border: none !important;
        padding: 14px !important;
        animation: slideInDown 0.5s ease-out !important;
    }}

    .dataframe td {{
        border-bottom: 1px solid {COLORS["border"]} !important;
        color: {COLORS["text_secondary"]} !important;
        padding: 12px 14px !important;
        transition: background-color 0.3s ease !important;
    }}

    .dataframe tr {{
        animation: fadeIn 0.4s ease-out !important;
    }}

    .dataframe tr:hover {{
        background-color: {COLORS["card_hover"]} !important;
    }}

    /* Expander */
    .streamlit-expanderHeader {{
        background-color: {COLORS["card_bg"]};
        border: 1px solid {COLORS["border"]};
        border-radius: 6px;
        padding: 14px 16px;
        transition: all 0.3s ease;
    }}

    .streamlit-expanderHeader:hover {{
        background-color: {COLORS["card_hover"]};
        border-color: {COLORS["primary"]};
    }}

    /* Detail Box */
    .detail-box {{
        background: linear-gradient(135deg, {COLORS["dark_bg"]} 0%, {COLORS["card_hover"]} 100%);
        border-radius: 6px;
        padding: 20px;
        margin-top: 20px;
        animation: slideInUp 0.5s ease-out 0.2s both;
        border: 1px solid {COLORS["border"]};
    }}

    /* Priority Guide Card */
    .priority-guide-card {{
        padding: 20px;
        background-color: {COLORS["card_bg"]};
        border-radius: 6px;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        cursor: pointer;
        position: relative;
    }}

    .priority-guide-card:hover {{
        transform: translateY(-8px);
        box-shadow: 0 12px 32px rgba(0, 0, 0, 0.3);
    }}

    .priority-guide-card::before {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 0px;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.05), transparent);
        transition: height 0.3s ease;
    }}

    .priority-guide-card:hover::before {{
        height: 100%;
    }}

    /* Footer */
    .footer {{
        text-align: center;
        padding: 40px 30px;
        margin-top: 60px;
        border-top: 1px solid {COLORS["border"]};
        color: {COLORS["text_muted"]};
        font-size: 0.9em;
        animation: slideInUp 0.6s ease-out;
    }}

    .footer p {{
        margin: 6px 0;
        animation: fadeIn 0.5s ease-out;
    }}

    /* Loading State */
    .loading {{
        background: linear-gradient(90deg, {COLORS["card_bg"]} 25%, {COLORS["card_hover"]} 50%, {COLORS["card_bg"]} 75%);
        background-size: 200% 100%;
        animation: shimmer 2s infinite;
    }}

    /* Responsive Design */
    @media (max-width: 1024px) {{
        .gov-header h1 {{
            font-size: 2.2em;
        }}
        
        .metric-card {{
            padding: 18px;
        }}

        .results-box {{
            padding: 30px;
        }}
    }}

    @media (max-width: 768px) {{
        .gov-header {{
            padding: 30px 20px;
        }}

        .gov-header h1 {{
            font-size: 1.8em;
        }}

        .gov-seal {{
            font-size: 2em;
        }}

        .metric-card {{
            margin-bottom: 15px;
            padding: 15px;
            font-size: 0.9em;
        }}

        .metric-value {{
            font-size: 1.8em;
        }}

        .sentiment-badge {{
            padding: 12px 20px;
            font-size: 1em;
        }}

        .section-title {{
            font-size: 1.3em;
            margin-top: 25px;
            margin-bottom: 18px;
        }}

        .input-container {{
            padding: 20px;
        }}

        .results-box {{
            padding: 20px;
        }}
    }}

    @media (max-width: 480px) {{
        .gov-header {{
            padding: 20px 15px;
        }}

        .gov-header h1 {{
            font-size: 1.4em;
            margin-bottom: 8px;
        }}

        .gov-header p {{
            font-size: 0.9em;
        }}

        .gov-seal {{
            font-size: 1.5em;
            margin-right: 10px;
        }}

        .status-bar {{
            flex-direction: column;
            align-items: flex-start;
            gap: 10px;
        }}

        .metric-value {{
            font-size: 1.5em;
        }}

        .stButton > button {{
            padding: 12px 30px;
            min-width: 140px;
            font-size: 0.9em;
        }}
    }}
</style>
""", unsafe_allow_html=True)

# HEADER SECTION
st.markdown("""
<div class="gov-header">
    <div style="font-size: 2.5em; display: inline-block; margin-right: 15px;">🏛️</div>
    <h1>Citizen Grievance Analysis Portal</h1>
    <p>Official Government System for Complaint Processing & Sentiment Analysis</p>
</div>
""", unsafe_allow_html=True)

# SIDEBAR
with st.sidebar:
    st.markdown(f'<div class="sidebar-title">System Control</div>', unsafe_allow_html=True)
    
    # Model Status
    model, vectorizer, metadata = load_models()
    if model is not None:
        st.markdown(f"""
        <div class="info-box success">
            <strong>Models:</strong> Loaded & Ready
        </div>
        """, unsafe_allow_html=True)
        
        with st.expander("Model Details"):
            st.write(f"**Model Type:** {metadata['model_type']}")
            st.write(f"**Vectorizer:** {metadata['vectorizer_type']}")
            st.write(f"**Test Accuracy:** {metadata['test_accuracy']:.1%}")
            st.write(f"**F1-Score (Macro):** {metadata['test_macro_f1']:.4f}")
            st.write(f"**Classes:** {len(metadata['sentiment_classes'])}")
    else:
        st.markdown(f"""
        <div class="info-box error">
            <strong>Models:</strong> Failed to Load
        </div>
        """, unsafe_allow_html=True)
    
    # API Status
    st.markdown(f'<div class="sidebar-title" style="margin-top: 25px;">API Status</div>', unsafe_allow_html=True)
    if test_api_connection():
        st.markdown(f"""
        <div class="info-box success">
            <strong>API:</strong> Connected<br>
            <small>FastAPI running on port 8000</small>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="info-box error">
            <strong>API:</strong> Disconnected<br>
            <small>Start with: python api/main.py</small>
        </div>
        """, unsafe_allow_html=True)
    
    # Instructions
    st.markdown(f'<div class="sidebar-title" style="margin-top: 25px;">How to Use</div>', unsafe_allow_html=True)
    st.write("""
    1. **Enter Complaint**
       Type or paste the citizen complaint text
    
    2. **Submit**
       Click "Analyze Complaint" button
    
    3. **Review Results**
       Check sentiment classification and priority level
    
    4. **Take Action**
       Route to appropriate department
    """)
    
    # System Info
    st.markdown(f'<div class="sidebar-title" style="margin-top: 25px;">System Information</div>', unsafe_allow_html=True)
    st.write(f"**Current Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    st.write(f"**Version:** 2.0.0 (Dark Theme)")
    st.write(f"**Framework:** Streamlit + FastAPI")

# STATUS BAR
st.markdown(f"""
<div class="status-bar">
    <div class="status-item">
        <div class="status-badge"></div>
        <span><strong>System Status:</strong> Operational</span>
    </div>
    <div class="status-item">
        <span><strong>Last Updated:</strong> {datetime.now().strftime('%H:%M:%S')}</span>
    </div>
</div>
""", unsafe_allow_html=True)

# MAIN CONTENT
if model is None or vectorizer is None:
    st.markdown(f"""
    <div class="info-box error">
        <strong>Error:</strong> Models not available. Please ensure trained models are in trained_models/ directory.
    </div>
    """, unsafe_allow_html=True)
else:
    # INPUT SECTION
    st.markdown('<div class="section-title">Submit Complaint</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="input-container">
        <label class="input-label">Citizen Complaint Description</label>
        <div class="input-help">Describe the issue or grievance in detail. The system will analyze sentiment and assign priority.</div>
    </div>
    """, unsafe_allow_html=True)
    
    complaint_text = st.text_area(
        "Complaint",
        value="The water main broke yesterday and caused flooding in the street affecting multiple households' access and property.",
        height=140,
        placeholder="Enter complaint text here...",
        label_visibility="collapsed"
    )
    
    # ANALYSIS BUTTON WITH ENHANCED STATE
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        analyze_button = st.button("🔍 Analyze Complaint", use_container_width=True, key="analyze_btn")
    
    if analyze_button:
        if complaint_text.strip():
            # Create a placeholder for analysis progress
            progress_placeholder = st.empty()
            result_placeholder = st.empty()
            
            # Simulate analysis with progress feedback
            with progress_placeholder.container():
                st.markdown('<div class="loading" style="height: 60px; border-radius: 6px; margin-bottom: 20px;"></div>', unsafe_allow_html=True)
            
            # Run analysis
            with st.spinner(""):
                result = predict_sentiment(complaint_text, model, vectorizer)
            
            time.sleep(0.3)  # Brief delay for visual effect
            progress_placeholder.empty()
            
            # ENHANCED RESULTS SECTION
            with result_placeholder.container():
                st.markdown('<div class="section-title">📊 Analysis Results</div>', unsafe_allow_html=True)
                
                st.markdown('<div class="results-box">', unsafe_allow_html=True)
                
                # Sentiment Badge with Animation
                sentiment = result["sentiment"]
                badge_style = f"""
                <div class="sentiment-badge" style="background: linear-gradient(135deg, {result['color']} 0%, {result['color']}dd 100%); color: white; position: relative; z-index: 2;">
                    <span style="font-size: 1.2em; margin-right: 12px;">{result['icon']}</span>
                    {result['priority_label']} · {sentiment}
                </div>
                """
                st.markdown(badge_style, unsafe_allow_html=True)
                
                # Metrics Grid - Enhanced
                st.markdown("""
                <div style="margin-bottom: 20px;">
                <p style="color: #C0C0C0; margin-bottom: 15px; font-weight: 500;">Key Performance Metrics</p>
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2, col3, col4 = st.columns(4, gap="medium")
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <span class="metric-label">📋 Classification</span>
                        <span class="metric-value">{sentiment}</span>
                        <span style="font-size: 0.75em; color: {COLORS['text_muted']};">Sentiment Type</span>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    priority_bar_color = result['color']
                    st.markdown(f"""
                    <div class="metric-card">
                        <span class="metric-label">⚡ Priority</span>
                        <span class="metric-value">{result['priority_score']}</span>
                        <div style="background-color: {COLORS['border']}; height: 4px; border-radius: 2px; margin-top: 8px; overflow: hidden;">
                            <div style="background: linear-gradient(90deg, {priority_bar_color}, {COLORS['accent']}); width: {(result['priority_score'] / 5) * 100}%; height: 100%; border-radius: 2px;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    level_text = result['priority_label']
                    level_color = result['color']
                    st.markdown(f"""
                    <div class="metric-card">
                        <span class="metric-label">🎯 Level</span>
                        <span class="metric-value" style="color: {level_color}; font-size: 1.5em;">{level_text[0]}</span>
                        <span style="font-size: 0.75em; color: {COLORS['text_muted']};">{level_text}</span>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    st.markdown(f"""
                    <div class="metric-card">
                        <span class="metric-label">⏱️ Status</span>
                        <span class="metric-value">✓</span>
                        <span style="font-size: 0.75em; color: {COLORS['success']};">Ready</span>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Details Section
                st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)
                st.markdown("""
                <div style="padding: 20px; background-color: """ + COLORS["dark_bg"] + """; border-radius: 6px; border-left: 4px solid """ + COLORS["primary"] + """; animation: slideInUp 0.5s ease-out;">
                    <h4 style="color: """ + COLORS["primary_light"] + """; margin-top: 0; margin-bottom: 12px; font-size: 1.05em;">Classification Details</h4>
                    <p style="color: """ + COLORS["text_secondary"] + """; margin: 8px 0; line-height: 1.6;">
                        <strong>Sentiment:</strong> This complaint has been classified as <span style="color: """ + COLORS["primary_light"] + """; font-weight: 600;">""" + sentiment + """</span> priority.
                    </p>
                    <p style="color: """ + COLORS["text_secondary"] + """; margin: 8px 0; line-height: 1.6;">
                        <strong>Description:</strong> """ + result['description'] + """
                    </p>
                    <p style="color: """ + COLORS["text_secondary"] + """; margin: 8px 0; line-height: 1.6;">
                        <strong>Action:</strong> Route to appropriate department and process according to priority level.
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Recommendation Box
                st.markdown("""
                <div style="margin-top: 25px; padding: 18px; background: linear-gradient(135deg, rgba(6, 168, 125, 0.1) 0%, rgba(10, 102, 194, 0.1) 100%); border-radius: 6px; border-left: 4px solid """ + COLORS["success"] + """; animation: slideInUp 0.6s ease-out 0.2s both;">
                    <p style="color: """ + COLORS["text_secondary"] + """; margin: 0; font-weight: 600; margin-bottom: 8px;">💡 Recommended Actions:</p>
                    <ul style="color: """ + COLORS["text_secondary"] + """; margin: 0; padding-left: 20px; line-height: 1.8;">
                        <li>Review and validate classification</li>
                        <li>Assign to relevant department</li>
                        <li>Set resolution timeline based on priority</li>
                        <li>Notify citizen of receipt</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Priority Classification Guide - Enhanced
                st.markdown('<div class="section-title">📚 Priority Classification Guide</div>', unsafe_allow_html=True)
                
                # Create a more visual priority guide with proper structure
                guide_html = '<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 18px; margin-bottom: 30px;">'
                
                delay_steps = [0, 0.1, 0.2, 0.3]  # Staggered delays
                delay_index = 0
                
                for sent, info in PRIORITY_MAPPING.items():
                    delay = delay_steps[delay_index] if delay_index < len(delay_steps) else 0
                    icon_delay = delay + 0.2
                    bar_delay = delay + 0.1
                    priority_width = (info['score'] / 5) * 100
                    
                    guide_html += f'<div style="padding: 20px; background-color: {COLORS["card_bg"]}; border-left: 5px solid {info["color"]}; border-radius: 6px; animation: scale 0.5s cubic-bezier(0.34, 1.56, 0.64, 1) {delay}s both; transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1); cursor: pointer;"><div style="font-size: 2em; margin-bottom: 10px; display: inline-block; animation: scale 0.6s ease-out {icon_delay}s both;">{info["icon"]}</div><div style="font-weight: 800; color: {COLORS["text_primary"]}; margin-bottom: 8px; font-size: 1.15em;">{info["label"]}</div><div style="font-size: 0.8em; color: {COLORS["text_muted"]}; margin-bottom: 10px; text-transform: uppercase; letter-spacing: 0.5px;">Priority Score</div><div style="background-color: {COLORS["dark_bg"]}; height: 6px; border-radius: 3px; overflow: hidden; margin-bottom: 12px;"><div style="background: linear-gradient(90deg, {info["color"]}, {COLORS["accent"]}); width: {priority_width}%; height: 100%; border-radius: 3px; animation: slideInLeft 0.8s ease-out {bar_delay}s both;"></div></div><div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;"><span style="color: {COLORS["text_muted"]}; font-size: 0.8em;">{info["score"]}/5</span><span style="color: {info["color"]}; font-weight: 700; font-size: 0.9em;">Level {info["score"]}</span></div><div style="height: 1px; background: linear-gradient(90deg, {info["color"]}, transparent); margin: 12px 0;"></div><div style="font-size: 0.85em; color: {COLORS["text_secondary"]}; line-height: 1.5; font-weight: 500;">{info["description"]}</div></div>'
                    delay_index += 1
                
                guide_html += '</div>'
                st.markdown(guide_html, unsafe_allow_html=True)
                
                # Comparison table
                guide_data = []
                for sent, info in PRIORITY_MAPPING.items():
                    guide_data.append({
                        "Priority": f"{info['icon']} {info['label']}",
                        "Score": f"{info['score']}/5",
                        "Description": info['description']
                    })
                
                guide_df = pd.DataFrame(guide_data)
                st.markdown("**Detailed Priority Reference Table:**")
                st.dataframe(guide_df, use_container_width=True, hide_index=True)
        else:
            st.markdown(f"""
            <div class="info-box warning">
                <strong>⚠️ Input Required</strong><br>
                <small>Please enter a complaint to analyze.</small>
            </div>
            """, unsafe_allow_html=True)

# MODEL INFORMATION SECTION - ENHANCED
with st.expander("📊 Model Information & Performance Metrics", expanded=False):
    # Create tabbed interface effect
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, """ + COLORS["card_bg"] + """ 0%, """ + COLORS["card_hover"] + """ 100%); padding: 25px; border-radius: 8px; border-left: 5px solid """ + COLORS["primary"] + """; animation: slideInLeft 0.5s ease-out;">
        <h4 style="color: """ + COLORS["primary_light"] + """; margin-top: 0; margin-bottom: 18px; font-size: 1.1em; text-transform: uppercase; letter-spacing: 1px;">⚙️ Model Specifications</h4>
        """, unsafe_allow_html=True)
        if metadata:
            spec_html = f"""
            <div style="display: grid; gap: 10px;">
                <div style="padding: 10px; background-color: {COLORS["dark_bg"]}; border-radius: 4px;">
                    <span style="color: {COLORS["text_muted"]}; font-size: 0.85em; text-transform: uppercase;">Model Type</span>
                    <div style="color: {COLORS["text_secondary"]}; font-weight: 600; margin-top: 4px;">{metadata['model_type']}</div>
                </div>
                <div style="padding: 10px; background-color: {COLORS["dark_bg"]}; border-radius: 4px;">
                    <span style="color: {COLORS["text_muted"]}; font-size: 0.85em; text-transform: uppercase;">Vectorizer</span>
                    <div style="color: {COLORS["text_secondary"]}; font-weight: 600; margin-top: 4px;">{metadata['vectorizer_type']}</div>
                </div>
                <div style="padding: 10px; background-color: {COLORS["dark_bg"]}; border-radius: 4px;">
                    <span style="color: {COLORS["text_muted"]}; font-size: 0.85em; text-transform: uppercase;">Max Features</span>
                    <div style="color: {COLORS["text_secondary"]}; font-weight: 600; margin-top: 4px;">{metadata['max_features']:,}</div>
                </div>
                <div style="padding: 10px; background-color: {COLORS["dark_bg"]}; border-radius: 4px;">
                    <span style="color: {COLORS["text_muted"]}; font-size: 0.85em; text-transform: uppercase;">N-gram Range</span>
                    <div style="color: {COLORS["text_secondary"]}; font-weight: 600; margin-top: 4px;">{str(metadata['ngram_range'])}</div>
                </div>
                <div style="padding: 10px; background-color: {COLORS["dark_bg"]}; border-radius: 4px;">
                    <span style="color: {COLORS["text_muted"]}; font-size: 0.85em; text-transform: uppercase;">Vocabulary Size</span>
                    <div style="color: {COLORS["text_secondary"]}; font-weight: 600; margin-top: 4px;">{metadata['vocabulary_size']:,}</div>
                </div>
            </div>
            """
            st.markdown(spec_html, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, {COLORS["card_bg"]} 0%, {COLORS["card_hover"]} 100%); padding: 25px; border-radius: 8px; border-left: 5px solid {COLORS["success"]}; animation: slideInRight 0.5s ease-out;">
        <h4 style="color: {COLORS["primary_light"]}; margin-top: 0; margin-bottom: 18px; font-size: 1.1em; text-transform: uppercase; letter-spacing: 1px;">📈 Performance Metrics</h4>
        """, unsafe_allow_html=True)
        if metadata:
            perf_html = f"""
            <div style="display: grid; gap: 10px;">
                <div style="padding: 15px; background: linear-gradient(135deg, rgba(6, 168, 125, 0.1) 0%, {COLORS["dark_bg"]} 100%); border-radius: 4px; border-left: 3px solid {COLORS["success"]};">
                    <span style="color: {COLORS["text_muted"]}; font-size: 0.85em; text-transform: uppercase;">Test Accuracy</span>
                    <div style="color: {COLORS["success"]}; font-weight: 700; margin-top: 6px; font-size: 1.4em;">{metadata['test_accuracy']:.1%}</div>
                </div>
                <div style="padding: 15px; background: linear-gradient(135deg, rgba(10, 102, 194, 0.1) 0%, {COLORS["dark_bg"]} 100%); border-radius: 4px; border-left: 3px solid {COLORS["primary"]};">
                    <span style="color: {COLORS["text_muted"]}; font-size: 0.85em; text-transform: uppercase;">F1-Score (Macro)</span>
                    <div style="color: {COLORS["primary_light"]}; font-weight: 700; margin-top: 6px; font-size: 1.4em;">{metadata['test_macro_f1']:.4f}</div>
                </div>
                <div style="padding: 15px; background: linear-gradient(135deg, rgba(255, 107, 53, 0.1) 0%, {COLORS["dark_bg"]} 100%); border-radius: 4px; border-left: 3px solid {COLORS["accent"]};">
                    <span style="color: {COLORS["text_muted"]}; font-size: 0.85em; text-transform: uppercase;">Classification Classes</span>
                    <div style="color: {COLORS["accent"]}; font-weight: 700; margin-top: 6px; font-size: 1.2em;">{len(metadata['sentiment_classes'])} Classes</div>
                </div>
                <div style="padding: 15px; background-color: {COLORS["dark_bg"]}; border-radius: 4px;">
                    <span style="color: {COLORS["text_muted"]}; font-size: 0.85em; text-transform: uppercase;">Classes Supported</span>
                    <div style="color: {COLORS["text_secondary"]}; font-weight: 600; margin-top: 6px; font-size: 0.95em;">{', '.join(metadata['sentiment_classes'])}</div>
                </div>
            </div>
            """
            st.markdown(perf_html, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Class Distribution - Enhanced Visualization
    st.markdown(f'<div class="section-title">📊 Sentiment Class Distribution</div>', unsafe_allow_html=True)
    if metadata:
        class_dist_data = []
        total = sum(metadata['class_distribution'].values())
        for sentiment, count in metadata['class_distribution'].items():
            percentage = (count / total) * 100
            class_dist_data.append({
                'Sentiment': sentiment,
                'Count': f"{count:,}",
                'Percentage': f"{percentage:.1f}%",
                'Proportion': percentage  # For visualization
            })
        
        # Create visual distribution
        dist_html = '<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 15px; margin-bottom: 30px;">'
        
        for item in class_dist_data:
            sentiment = item['Sentiment']
            count = int(item['Count'].replace(',', ''))
            percentage = item['Proportion']
            
            # Get color for sentiment
            color = PRIORITY_MAPPING.get(sentiment, {}).get('color', COLORS["primary"])
            
            dist_html += f"""
            <div style="padding: 18px; background-color: {COLORS["card_bg"]}; border-radius: 8px; border-left: 4px solid {color}; animation: scale 0.5s ease-out; transition: all 0.3s ease;">
                <div style="font-weight: 700; color: {COLORS["text_primary"]}; margin-bottom: 12px; font-size: 1.05em;">{sentiment}</div>
                <div style="background-color: {COLORS["dark_bg"]}; height: 8px; border-radius: 4px; overflow: hidden; margin-bottom: 12px;">
                    <div style="background: linear-gradient(90deg, {color}, {COLORS["accent"]}); width: {percentage}%; height: 100%; border-radius: 4px; transition: width 0.8s ease-out;"></div>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="color: {COLORS["text_muted"]}; font-size: 0.85em;">{count:,} samples</span>
                    <span style="color: {color}; font-weight: 700;">{percentage:.1f}%</span>
                </div>
            </div>
            """
        
        dist_html += '</div>'
        st.markdown(dist_html, unsafe_allow_html=True)
        
        # Data table
        dist_df = pd.DataFrame(class_dist_data)[['Sentiment', 'Count', 'Percentage']]
        st.dataframe(dist_df, use_container_width=True, hide_index=True)

# FOOTER - Enhanced
st.markdown("""
<div class="footer">
    <p style="font-size: 1.05em; font-weight: 600; color: """ + COLORS["primary_light"] + """; margin-bottom: 12px;">
        🏛️ Official Citizen Grievance Analysis Portal
    </p>
    <p>Professional Government System for Complaint Processing & Sentiment Analysis</p>
    <p style="color: """ + COLORS["text_muted"] + """; font-size: 0.85em; margin-top: 12px;">
        Built with FastAPI, Streamlit, and Advanced Machine Learning
    </p>
    <p style="color: """ + COLORS["text_muted"] + """; font-size: 0.8em; margin-top: 8px;">
        © 2026 Government Services. All rights reserved.
    </p>
</div>
""", unsafe_allow_html=True)
