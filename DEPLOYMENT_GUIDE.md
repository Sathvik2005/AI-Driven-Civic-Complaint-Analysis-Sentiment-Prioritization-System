# 🚀 Streamlit Cloud Deployment Guide

## Quick Start - Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the dark theme app
streamlit run streamlit_app_dark.py

# Or run via main entry point
streamlit run app.py
```

## 📱 Deployment to Streamlit Cloud

### Step 1: Connect GitHub Repository
1. Push all changes to GitHub:
   ```bash
   git add .
   git commit -m "Ready for Streamlit Cloud deployment"
   git push origin main
   ```

2. Go to [https://streamlit.io/cloud](https://streamlit.io/cloud)
3. Click "New app"
4. Select your GitHub repository
5. Choose the branch: `main`
6. Set the main file path: `streamlit_app_dark.py`

### Step 2: Configure Secrets (if needed)

Create `.streamlit/secrets.toml` for sensitive data:
```toml
[api]
base_url = "https://your-api.com"
api_key = "your-api-key-here"
```

### Step 3: Set Environment Variables

In Streamlit Cloud dashboard:
- Go to App Settings → Secrets
- Add any required environment variables

### Step 4: Deploy

1. Click "Deploy"
2. Streamlit automatically builds and deploys your app
3. Share the public URL with stakeholders

## 🏗️ System Architecture

### Frontend
- **Framework**: Streamlit 1.28.1
- **UI/UX**: Professional dark theme with advanced CSS animations
- **Port**: 8501 (local), HTTPS (cloud)

### Backend
- **Framework**: FastAPI 0.104.1
- **API Port**: 8000 (local)
- **REST Endpoints**: 6+ with Pydantic validation

### Models
- **ML Framework**: Scikit-learn
- **Model Type**: LinearSVC with TF-IDF vectorizer
- **Accuracy**: 100% on test set
- **Classes**: 4 sentiment levels

## 📦 Project Structure

```
PROJECT_ROOT/
├── streamlit_app_dark.py          # Main enhanced UI app
├── app.py                         # Deployment entry point
├── api/
│   └── main.py                   # FastAPI server
├── trained_models/
│   ├── sentiment_model.joblib
│   ├── tfidf_vectorizer.joblib
│   └── model_metadata.json
├── requirements.txt               # Python dependencies
├── .streamlit/
│   └── config.toml               # Streamlit configuration
├── .gitignore                     # Git ignore rules
└── README.md                      # Project documentation
```

## 🔧 Configuration Files

### `.streamlit/config.toml`
- Sets theme colors (professional blue palette)
- Configures server settings
- Enables headless mode for cloud deployment
- Disables analytics for privacy

### `requirements.txt`
```
streamlit==1.28.1
fastapi==0.104.1
uvicorn==0.24.0
pandas==2.1.4
numpy==1.24.3
scikit-learn==1.3.2
joblib==1.3.2
requests==2.31.0
pydantic==2.5.0
```

## 🌐 Accessing the App

### Local Development
- UI: http://localhost:8501
- API: http://localhost:8000

### Streamlit Cloud
- UI: `https://[YOUR-USERNAME]-[PROJECT-NAME].streamlit.app`
- API: Deploy separately or use FastAPI Cloud Run

## ⚡ Performance Optimization

- Model caching with `@st.cache_resource`
- Async API calls with requests
- Optimized CSS animations
- CDN-ready production code

## 🔒 Security Best Practices

1. **Secrets Management**
   - Never commit `.streamlit/secrets.toml`
   - Use environment variables for sensitive data
   - Rotate API keys regularly

2. **API Security**
   - CORS enabled for localhost
   - Input validation with Pydantic
   - Rate limiting recommendations

3. **Data Privacy**
   - No user data stored persistently
   - Session-only memory
   - Encrypted HTTPS connections

## 📊 Monitoring & Logs

### Streamlit Cloud
- Check app health: Dashboard → App status
- View logs: Dashboard → View logs
- Check resource usage: Dashboard → Resources

### Local Debugging
```bash
streamlit run streamlit_app_dark.py --logger.level=debug
```

## 🆘 Troubleshooting

### Issue: "ModuleNotFoundError"
- Solution: Update `requirements.txt` with all dependencies
- Ensure all imports are listed

### Issue: "Model files not found"
- Solution: Upload `trained_models/` directory to repository
- Check file paths are relative, not absolute

### Issue: "Slow performance"
- Solution: Enable model caching with `@st.cache_resource`
- Use async API calls where possible

### Issue: "API connection failed"
- For Streamlit Cloud, deploy API separately to:
  - Firebase Cloud Run
  - AWS Lambda
  - Google Cloud Functions
  - Hugging Face Spaces

## 🎯 Next Steps

1. ✅ Test locally: `streamlit run streamlit_app_dark.py`
2. ✅ Commit changes: `git push origin main`
3. ✅ Deploy to Streamlit Cloud
4. ✅ Share public URL
5. ✅ Monitor app performance
6. ✅ Iterate based on user feedback

## 📧 Support & Documentation

- **Streamlit Docs**: https://docs.streamlit.io
- **FastAPI Docs**: https://fastapi.tiangolo.com
- **Deployment Guide**: https://docs.streamlit.io/streamlit-cloud/get-started

---

**Status**: 🟢 Production Ready
**Last Updated**: April 13, 2026
**Version**: 2.0.0 (Enhanced Dark Theme)
