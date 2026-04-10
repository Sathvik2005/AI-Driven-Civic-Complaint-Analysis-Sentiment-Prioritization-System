# WEEK 3 - QUICK START GUIDE

## Overview

This guide provides the fastest way to get the Citizen Grievance Analysis System running with your **original NYC 311 dataset**.

---

## ⚡ Three Launch Methods

### Method 1: Windows Users (Recommended) - One Click

```bash
# Simply double-click this file:
START_PROJECT.bat
```

**What happens:**
- Dependencies installed automatically
- Dataset downloaded (if needed)
- Models trained (if needed - takes 2-5 min first time)
- FastAPI opens on http://localhost:8000
- Streamlit opens on http://localhost:8501
- Both services run in separate windows

**Uninstall:** Just close both windows

---

### Method 2: Python Users - Single Command

```bash
# Run from project root directory
python run_project.py
```

**What happens:**
- Same as Method 1, all in one process
- View all logs in terminal
- Press Ctrl+C to stop everything

---

### Method 3: Manual - Full Control (Advanced)

```bash
# Terminal 1: Start API
python -m uvicorn api.main:app --reload --port 8000

# Terminal 2: Start UI
python -m streamlit run streamlit_app.py --server.port 8501
```

**When to use:** Development, debugging, or running services on different machines

---

## 🎯 Usage Examples

### Example 1: Single Complaint Analysis

**Input:**
```
"Water leak in community center building, immediate action needed"
```

**Output:**
```
Department: DEPARTMENT OF ENVIRONMENTAL PROTECTION
Sentiment: Critical
Priority Score: 95/100
Confidence: 94%
```

---

### Example 2: Batch Processing (CSV)

**Step 1:** Prepare CSV with column `complaint_text`:
```csv
complaint_text
"Pothole on 5th Avenue making roads unsafe"
"Broken street light near park entrance"
"Noise complaint from construction at night"
```

**Step 2:** Upload in Streamlit UI (Batch Processing tab)

**Step 3:** Download results with predictions:
```csv
complaint_text,predicted_department,sentiment,priority_score
"Pothole...",DEPARTMENT OF TRANSPORTATION,Negative,78
...
```

---

### Example 3: System Information

In Streamlit UI, click **"System Info"** tab to see:
- ✓ Model status (trained/not trained)
- ✓ Dataset information (rows, features)
- ✓ Sentiment labels guide
- ✓ API documentation links
- ✓ Performance metrics

---

## 📊 Performance Notes

### First Run (Training)
- **Duration:** 2-5 minutes
- **GPU:** Not required (uses CPU)
- **RAM:** 4GB minimum
- **What happens:** 
  - Loads 50K+ NYC 311 records
  - Trains TF-IDF vectorizer (5000 features)
  - Trains sentiment classifier (LinearSVC)
  - Saves models to `trained_models/`

### Subsequent Runs
- **Duration:** <5 seconds startup
- **Models loaded:** From disk cache
- **API ready:** http://localhost:8000 (30 sec)
- **UI ready:** http://localhost:8501 (5 sec)

### Batch Processing Speed
- **Single prediction:** ~50ms (network latency included)
- **100 complaints:** ~8 seconds (API + UI rendering)
- **1000 complaints:** ~80 seconds
- **Bottleneck:** TF-IDF vectorization and model inference

---

## 🔧 Configuration & Customization

### Change API Port (default: 8000)

**Method 1 (Windows batch file):**
```batch
REM In START_PROJECT.bat, change:
python -m uvicorn api.main:app --reload --port 9000
```

**Method 2 (Manual):**
```bash
python -m uvicorn api.main:app --reload --port 9000
```

---

### Change Streamlit Port (default: 8501)

**Method 1 (Windows batch file):**
```batch
REM In START_PROJECT.bat, change:
python -m streamlit run streamlit_app.py --server.port 9501
```

**Method 2 (Manual):**
```bash
python -m streamlit run streamlit_app.py --server.port 9501
```

---

### Use Different Dataset

**Option A: Replace with local CSV**
```bash
# Put your CSV in:
datasets/my_dataset.csv

# Code will auto-detect and use it
```

**Option B: Change dataset in code**

Edit `WEEK3_IMPROVED_ORIGINAL_DATASET.ipynb`, Cell 4:
```python
# Change from:
df = load_nyc_311_data()

# To:
df = pd.read_csv("path/to/your/dataset.csv")
# Make sure columns include: complaint_text, agency
```

Then retrain:
```bash
python -m jupyter nbconvert --to notebook --execute \
    WEEK3_IMPROVED_ORIGINAL_DATASET.ipynb
```

---

## ❓ Troubleshooting

### Issue: "Port already in use"

**Solution:**
```bash
# Find and kill process on port 8000
# Windows:
netstat -ano | findstr ":8000"
taskkill /PID <PID> /F

# Or use different port:
python -m uvicorn api.main:app --port 9000
```

---

### Issue: "No module named sklearn"

**Solution:**
```bash
pip install -r requirements.txt
```

---

### Issue: "Dataset not found"

**Solution:**
```bash
# Manually download
python download_dataset.py

# Or re-train to regenerate:
python -m jupyter nbconvert --to notebook --execute \
    WEEK3_IMPROVED_ORIGINAL_DATASET.ipynb
```

---

### Issue: "Models not trained"

**Solution:**
```bash
# Run training notebook
python -m jupyter nbconvert --to notebook --execute \
    "WEEK3_IMPROVED_ORIGINAL_DATASET.ipynb"

# Or use run_project.py
python run_project.py
```

---

## 📚 Sentiment Labels Guide

| Label | Meaning | Score Range |
|-------|---------|-------------|
| **Critical** | Immediate action required | 80-100 |
| **Negative** | Significant issue, urgent | 60-79 |
| **Neutral** | Routine complaint | 40-59 |
| **Positive** | Minor/maintenance only | 0-39 |

---

## 🚀 Next Steps

1. **Test Single Mode**: Try a few manual predictions
2. **Test Batch Mode**: Upload sample CSV file
3. **Check System Info**: Verify models are working
4. **Review API Docs**: Visit http://localhost:8000/docs
5. **Modify for your use case**: Edit datasets/code as needed

---

## 📞 Support

For detailed documentation, see:
- `WEEK3_DEPLOYMENT_GUIDE.md` - Technical deep dive
- `WEEK3_COMPLETION_SUMMARY.md` - What was delivered
- `api/main.py` - API code documentation
- `streamlit_app.py` - UI code documentation

---

## 🎓 Learning Path

**Week 1:** Data Collection & Cleaning (See `Week1_*.ipynb`)
- Load NYC 311 data
- Clean and preprocess text
- Exploratory data analysis

**Week 2:** Department Routing (See `Week2_*.ipynb`)
- Build department classifier
- TF-IDF vectorization
- Model evaluation

**Week 3:** Sentiment Analysis (This folder)
- Sentiment classification
- Priority scoring
- Deployment pipeline

**Week 4:** REST API + Web UI (This folder)
- FastAPI backend
- Streamlit frontend
- Production deployment

---

## 📝 Version Info

- **Model Type:** LinearSVC + TF-IDF
- **Dataset:** NYC 311 Service Requests (50K+ records)
- **Language:** Python 3.8+
- **Framework:** Streamlit + FastAPI
- **Last Updated:** 2024
- **Status:** Production Ready ✓

---

## 💡 Pro Tips

1. **Batch processing** is faster than single predictions for >10 complaints
2. **System Info tab** shows useful debugging information
3. **Model retraining** happens automatically if source data changes
4. **API is stateless** - safe to stop/restart anytime
5. **Streamlit auto-refreshes** on Python file changes (great for development!)

---

**Ready to start?** Pick your method above and get running! 🚀
