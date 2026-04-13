from __future__ import annotations

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any
import requests

import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
SENTIMENT_MODEL_PATH = BASE_DIR / "trained_models" / "sentiment_model.joblib"
VECTORIZER_MODEL_PATH = BASE_DIR / "trained_models" / "tfidf_vectorizer.joblib"
METADATA_PATH = BASE_DIR / "trained_models" / "model_metadata.json"
API_BASE_URL = "http://localhost:8000"

SENTIMENT_COLORS = {
	"Critical": "#d32f2f",
	"Negative": "#f57c00",
	"Neutral": "#558b2f",
	"Positive": "#0097a7"
}

PRIORITY_MAPPING = {
	"Critical": {"score": 5, "label": "URGENT", "description": "Requires immediate action"},
	"Negative": {"score": 4, "label": "HIGH", "description": "Should be addressed soon"},
	"Neutral": {"score": 3, "label": "MEDIUM", "description": "Standard processing"},
	"Positive": {"score": 2, "label": "LOW", "description": "Routine handling"}
}


@st.cache_resource
def load_models():
	"""Load pre-trained sentiment model and vectorizer"""
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
			"color": SENTIMENT_COLORS.get(prediction, "#666666")
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
	page_title="Citizen Grievance Analysis System",
	page_icon="📋",
	layout="wide",
	initial_sidebar_state="expanded"
)

# CUSTOM CSS FOR PROFESSIONAL STYLING
st.markdown("""
<style>
	:root {
		--primary-color: #1565c0;
		--secondary-color: #0097a7;
		--danger-color: #d32f2f;
		--warning-color: #f57c00;
		--success-color: #01579b;
		--text-dark: #2c3e50;
		--text-light: #ecf0f1;
		--border-color: #bdc3c7;
		--bg-light: #f8f9fa;
	}
	
	* {
		font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
	}
	
	/* Main container */
	.main {
		background-color: #ffffff;
	}
	
	/* Header styling */
	.header-section {
		background: linear-gradient(135deg, #1565c0 0%, #0097a7 100%);
		color: white;
		padding: 40px 30px;
		border-radius: 10px;
		margin-bottom: 30px;
		box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
	}
	
	.header-section h1 {
		margin: 0;
		font-size: 2.5em;
		font-weight: 700;
		letter-spacing: -0.5px;
	}
	
	.header-section p {
		margin: 10px 0 0 0;
		font-size: 1.1em;
		opacity: 0.95;
	}
	
	/* Section headers */
	.section-header {
		color: #1565c0;
		font-size: 1.5em;
		font-weight: 600;
		margin-top: 25px;
		margin-bottom: 15px;
		border-bottom: 3px solid #0097a7;
		padding-bottom: 10px;
	}
	
	/* Input area */
	.input-section {
		background-color: #f8f9fa;
		padding: 20px;
		border-radius: 8px;
		border-left: 4px solid #1565c0;
		margin-bottom: 20px;
	}
	
	/* Results container */
	.results-container {
		background: linear-gradient(to right, #f8f9fa 0%, #ffffff 100%);
		padding: 25px;
		border-radius: 8px;
		border: 1px solid #e0e0e0;
		margin-top: 20px;
	}
	
	/* Sentiment badge */
	.sentiment-badge {
		display: inline-block;
		padding: 12px 24px;
		border-radius: 25px;
		font-weight: 700;
		font-size: 1.1em;
		text-transform: uppercase;
		letter-spacing: 1px;
		box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
	}
	
	.sentiment-critical {
		background-color: #d32f2f;
		color: white;
	}
	
	.sentiment-negative {
		background-color: #f57c00;
		color: white;
	}
	
	.sentiment-neutral {
		background-color: #558b2f;
		color: white;
	}
	
	.sentiment-positive {
		background-color: #0097a7;
		color: white;
	}
	
	/* Metrics styling */
	.metric-card {
		background: white;
		padding: 20px;
		border-radius: 8px;
		border: 1px solid #e0e0e0;
		box-shadow: 0 2px 4px rgba(0, 0, 0, 0.08);
		text-align: center;
	}
	
	.metric-label {
		color: #666666;
		font-size: 0.95em;
		font-weight: 500;
		margin-bottom: 8px;
		text-transform: uppercase;
		letter-spacing: 0.5px;
	}
	
	.metric-value {
		color: #1565c0;
		font-size: 2em;
		font-weight: 700;
	}
	
	/* Button styling */
	.stButton > button {
		background-color: #1565c0;
		color: white;
		font-weight: 600;
		padding: 12px 40px;
		border: none;
		border-radius: 6px;
		font-size: 1em;
		transition: all 0.3s ease;
		box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
	}
	
	.stButton > button:hover {
		background-color: #0d47a1;
		box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
		transform: translateY(-2px);
	}
	
	.stButton > button:active {
		transform: translateY(0);
	}
	
	/* Text area styling */
	.stTextArea > textarea {
		border: 2px solid #e0e0e0;
		border-radius: 6px;
		font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
		font-size: 1em;
		padding: 15px;
		transition: border-color 0.3s ease;
	}
	
	.stTextArea > textarea:focus {
		border-color: #1565c0;
		box-shadow: 0 0 0 3px rgba(21, 101, 192, 0.1);
	}
	
	/* Info boxes */
	.info-box {
		background-color: #e3f2fd;
		border-left: 4px solid #1565c0;
		padding: 15px;
		border-radius: 6px;
		margin: 15px 0;
		color: #1565c0;
		font-size: 0.95em;
	}
	
	.success-box {
		background-color: #e8f5e9;
		border-left: 4px solid #558b2f;
		padding: 15px;
		border-radius: 6px;
		margin: 15px 0;
	}
	
	.warning-box {
		background-color: #fff3e0;
		border-left: 4px solid #f57c00;
		padding: 15px;
		border-radius: 6px;
		margin: 15px 0;
	}
	
	/* Data table styling */
	.dataframe {
		width: 100%;
		border-collapse: collapse;
		font-size: 0.95em;
	}
	
	.dataframe th {
		background-color: #1565c0;
		color: white;
		padding: 12px;
		text-align: left;
		font-weight: 600;
	}
	
	.dataframe td {
		padding: 12px;
		border-bottom: 1px solid #e0e0e0;
	}
	
	.dataframe tr:hover {
		background-color: #f8f9fa;
	}
	
	/* Sidebar styling */
	.sidebar .sidebar-content {
		padding: 20px;
	}
	
	.sidebar-header {
		font-size: 1.3em;
		font-weight: 700;
		color: #1565c0;
		margin-bottom: 15px;
		padding-bottom: 10px;
		border-bottom: 2px solid #0097a7;
	}
	
	/* Footer */
	.footer {
		text-align: center;
		padding: 20px;
		color: #999999;
		font-size: 0.9em;
		margin-top: 40px;
		border-top: 1px solid #e0e0e0;
	}
	
	/* Responsive design */
	@media (max-width: 768px) {
		.header-section h1 {
			font-size: 1.8em;
		}
		
		.metric-card {
			margin-bottom: 15px;
		}
	}
</style>
""", unsafe_allow_html=True)

# HEADER SECTION
st.markdown("""
<div class="header-section">
	<h1>Citizen Grievance Analysis System</h1>
	<p>AI-Powered Sentiment Analysis and Priority Classification</p>
</div>
""", unsafe_allow_html=True)

# SIDEBAR
with st.sidebar:
	st.markdown('<div class="sidebar-header">System Status</div>', unsafe_allow_html=True)
	
	# Model status
	model, vectorizer, metadata = load_models()
	if model is not None:
		st.success("Models Loaded Successfully")
		with st.expander("Model Details"):
			st.write(f"**Model Type:** {metadata['model_type']}")
			st.write(f"**Vectorizer:** {metadata['vectorizer_type']}")
			st.write(f"**Test Accuracy:** {metadata['test_accuracy']:.1%}")
			st.write(f"**Sentiment Classes:** {', '.join(metadata['sentiment_classes'])}")
	else:
		st.error("Failed to load models")
	
	# API status
	st.markdown("---")
	if test_api_connection():
		st.success("API Connected")
		st.caption("FastAPI server is running")
	else:
		st.warning("API Not Available")
		st.caption("Start API with: python api/main.py")
	
	# Instructions
	st.markdown("---")
	st.markdown('<div class="sidebar-header">How to Use</div>', unsafe_allow_html=True)
	st.write("""
	1. Enter or paste a complaint text
	2. Click 'Analyze Complaint'
	3. View sentiment classification
	4. Check priority score
	5. Review confidence metrics
	""")

# MAIN CONTENT
if model is None or vectorizer is None:
	st.markdown('<div class="warning-box"><strong>Error:</strong> Models not available. Please ensure trained models are in trained_models/ directory.</div>', unsafe_allow_html=True)
else:
	# Input Section
	st.markdown('<div class="section-header">Enter Complaint</div>', unsafe_allow_html=True)
	
	st.markdown('<div class="input-section">', unsafe_allow_html=True)
	complaint_text = st.text_area(
		"Complaint Description",
		value="The streetlights in our lane have been out for three days and the road feels unsafe at night. Please send someone to fix them urgently.",
		height=120,
		placeholder="Enter the complaint text here..."
	)
	st.markdown('</div>', unsafe_allow_html=True)
	
	# Analyze Button
	col1, col2 = st.columns([1, 4])
	with col1:
		analyze_button = st.button("Analyze Complaint", use_container_width=True)
	
	# Results
	if analyze_button and complaint_text.strip():
		with st.spinner("Analyzing complaint..."):
			result = predict_sentiment(complaint_text, model, vectorizer)
		
		# Results Display
		st.markdown('<div class="results-container">', unsafe_allow_html=True)
		
		st.markdown('<div class="section-header">Analysis Results</div>', unsafe_allow_html=True)
		
		# Sentiment Badge
		sentiment = result["sentiment"]
		sentiment_class = f"sentiment-{sentiment.lower()}"
		st.markdown(f"""
		<div class="sentiment-badge {sentiment_class}">
			{result["priority_label"]} - {sentiment}
		</div>
		""", unsafe_allow_html=True)
		
		# Metrics Row
		st.markdown("---")
		col1, col2, col3, col4 = st.columns(4)
		
		with col1:
			st.markdown("""
			<div class="metric-card">
				<div class="metric-label">Sentiment Classification</div>
				<div class="metric-value" style="color: {};">{}</div>
			</div>
			""".format(result["color"], sentiment), unsafe_allow_html=True)
		
		with col2:
			st.markdown("""
			<div class="metric-card">
				<div class="metric-label">Priority Level</div>
				<div class="metric-value">{}/{}</div>
			</div>
			""".format(result["priority_score"], 5), unsafe_allow_html=True)
		
		with col3:
			st.markdown("""
			<div class="metric-card">
				<div class="metric-label">Priority Label</div>
				<div class="metric-value" style="font-size: 1.3em;">{}</div>
			</div>
			""".format(result["priority_label"]), unsafe_allow_html=True)
		
		with col4:
			classification_map = {
				"Critical": "Urgent Action Required",
				"Negative": "Prompt Response Needed",
				"Neutral": "Standard Processing",
				"Positive": "Routine Handling"
			}
			st.markdown("""
			<div class="metric-card">
				<div class="metric-label">Action Required</div>
				<div class="metric-value" style="font-size: 0.9em; color: #1565c0;">{}</div>
			</div>
			""".format(classification_map.get(sentiment, "N/A")), unsafe_allow_html=True)
		
		# Detailed Information
		st.markdown("---")
		st.markdown('<div class="section-header">Classification Details</div>', unsafe_allow_html=True)
		
		details_df = pd.DataFrame({
			"Metric": ["Sentiment", "Priority Score", "Classification", "Processing Time"],
			"Value": [
				f"{sentiment} ({result['color']})",
				f"{result['priority_score']}/5",
				result["priority_label"],
				"Real-time"
			]
		})
		
		st.dataframe(details_df, use_container_width=True, hide_index=True)
		
		# Color Legend
		st.markdown("---")
		st.markdown('<div class="section-header">Sentiment Color Guide</div>', unsafe_allow_html=True)
		
		legend_cols = st.columns(4)
		sentiments_list = ["Critical", "Negative", "Neutral", "Positive"]
		colors_list = ["#d32f2f", "#f57c00", "#558b2f", "#0097a7"]
		descriptions = [
			"Requires immediate action",
			"Should be addressed soon",
			"Standard processing needed",
			"Routine handling"
		]
		
		for idx, (sent, color, desc) in enumerate(zip(sentiments_list, colors_list, descriptions)):
			with legend_cols[idx]:
				st.markdown(f"""
				<div style="
					background-color: {color};
					color: white;
					padding: 12px;
					border-radius: 6px;
					text-align: center;
					font-weight: 600;
					margin-bottom: 8px;
				">
					{sent}
				</div>
				<p style="font-size: 0.85em; text-align: center; color: #666;">{desc}</p>
				""", unsafe_allow_html=True)
		
		st.markdown('</div>', unsafe_allow_html=True)
		
	elif analyze_button:
		st.markdown('<div class="warning-box">Please enter a complaint text to analyze.</div>', unsafe_allow_html=True)

# MODEL INFORMATION SECTION
with st.expander("Model Information & Performance"):
	if metadata:
		col1, col2 = st.columns(2)
		
		with col1:
			st.markdown("**Model Specifications**")
			st.write(f"Model Type: {metadata['model_type']}")
			st.write(f"Vectorizer: {metadata['vectorizer_type']}")
			st.write(f"Max Features: {metadata['max_features']}")
			st.write(f"N-gram Range: {metadata['ngram_range']}")
			st.write(f"Vocabulary Size: {metadata['vocabulary_size']}")
		
		with col2:
			st.markdown("**Performance Metrics**")
			st.write(f"Test Accuracy: {metadata['test_accuracy']:.1%}")
			st.write(f"Test Macro F1-Score: {metadata['test_macro_f1']:.4f}")
			st.write(f"Classes: {', '.join(metadata['sentiment_classes'])}")
		
		st.markdown("**Class Distribution**")
		class_dist_df = pd.DataFrame(
			list(metadata['class_distribution'].items()),
			columns=['Sentiment', 'Count']
		)
		class_dist_df['Percentage'] = (class_dist_df['Count'] / class_dist_df['Count'].sum() * 100).round(2)
		st.dataframe(class_dist_df, use_container_width=True, hide_index=True)

# FOOTER
st.markdown("""
<div class="footer">
	<p>Citizen Grievance Analysis System | Production-Ready | Resume Portfolio Project</p>
	<p>Built with Streamlit, Scikit-learn, and FastAPI | NYC 311 Dataset</p>
</div>
""", unsafe_allow_html=True)
