from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.inference import load_model_bundle, predict_department, predict_sentiment


BASE_DIR = Path(__file__).resolve().parent
DEPARTMENT_MODEL_PATH = BASE_DIR / "trained_models" / "department_model.joblib"
SENTIMENT_MODEL_PATH = BASE_DIR / "trained_models" / "sentiment_model.joblib"


@st.cache_resource
def load_artifacts() -> tuple[dict, dict]:
	return load_model_bundle(DEPARTMENT_MODEL_PATH), load_model_bundle(SENTIMENT_MODEL_PATH)


st.set_page_config(page_title="Citizen Grievance Demo", layout="wide")
st.title("AI-Driven Citizen Grievance and Sentiment Analysis")
st.write(
	"This demo predicts the likely department for a complaint and assigns a sentiment-based priority score."
)

if not DEPARTMENT_MODEL_PATH.exists() or not SENTIMENT_MODEL_PATH.exists():
	st.warning("Model artifacts are missing. Run the Week 2 and Week 3 training notebooks first.")
else:
	department_bundle, sentiment_bundle = load_artifacts()
	complaint_text = st.text_area(
		"Enter a citizen complaint",
		value="Streetlights in our lane have been out for three days and the road feels unsafe at night.",
		height=180,
	)

	if st.button("Analyze Complaint"):
		department = predict_department(complaint_text, department_bundle)
		sentiment_result = predict_sentiment(complaint_text, sentiment_bundle)

		col1, col2, col3 = st.columns(3)
		col1.metric("Predicted Department", department)
		col2.metric("Predicted Sentiment", sentiment_result["sentiment"])
		col3.metric("Priority Score", sentiment_result["priority_score"])
