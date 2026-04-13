"""
Main Streamlit Application Entry Point
Citizen Grievance Analysis System - Production Deployment
"""

import streamlit as st
import os
import sys

# Set page config early
st.set_page_config(
    page_title="Citizen Grievance Analysis Portal",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import the dark theme app as the main interface
# This ensures compatibility with Streamlit Cloud deployment
from streamlit_app_dark import *

if __name__ == "__main__":
    pass
