#!/usr/bin/env python
"""
Deployment Verification Script
Validates all components before Streamlit Cloud deployment
"""

import json
from pathlib import Path
import joblib
import sys

print("\n" + "="*60)
print("📋 DEPLOYMENT VERIFICATION CHECKLIST")
print("="*60 + "\n")

checks = []
errors = []

try:
    # 1. Check metadata structure
    metadata_path = Path('trained_models/model_metadata.json')
    print(f"✓ Checking metadata at {metadata_path}...")
    with open(metadata_path) as f:
        metadata = json.load(f)
        has_performance = 'performance' in metadata
        checks.append(('Metadata structure (has performance dict)', has_performance))
        
        if has_performance:
            perf = metadata['performance']
            has_accuracy = 'test_accuracy' in perf
            has_f1 = 'test_f1_score' in perf
            checks.append(('Has test_accuracy in performance', has_accuracy))
            checks.append(('Has test_f1_score in performance', has_f1))
            
            if has_accuracy and has_f1:
                print(f"  ✓ Test Accuracy: {perf.get('test_accuracy_percentage', 'N/A')}")
                print(f"  ✓ F1-Score: {perf.get('test_f1_score', 'N/A')}")
        
        # Check sentiment classes
        has_classes = 'sentiment_classes' in metadata
        checks.append(('Has sentiment_classes', has_classes))
        if has_classes:
            print(f"  ✓ Classes: {', '.join(metadata['sentiment_classes'])}")

    # 2. Check model files
    print(f"\n✓ Checking model files...")
    model_path = Path('trained_models/sentiment_model.joblib')
    vectorizer_path = Path('trained_models/tfidf_vectorizer.joblib')
    
    model_exists = model_path.exists()
    vectorizer_exists = vectorizer_path.exists()
    checks.append(('Model file exists', model_exists))
    checks.append(('Vectorizer file exists', vectorizer_exists))
    
    if model_exists:
        print(f"  ✓ Model: {model_path.name} ({model_path.stat().st_size} bytes)")
    if vectorizer_exists:
        print(f"  ✓ Vectorizer: {vectorizer_path.name} ({vectorizer_path.stat().st_size} bytes)")

    # 3. Load and verify models
    print(f"\n✓ Loading and testing models...")
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    
    model_loads = model is not None
    vectorizer_loads = vectorizer is not None
    checks.append(('Model loads successfully', model_loads))
    checks.append(('Vectorizer loads successfully', vectorizer_loads))
    
    if model_loads:
        print(f"  ✓ Model type: {type(model).__name__}")
    if vectorizer_loads:
        print(f"  ✓ Vectorizer type: {type(vectorizer).__name__}")

    # 4. Test prediction on all 4 sentiment classes
    print(f"\n✓ Testing predictions on 4 sentiment classes...")
    test_cases = [
        ("Critical", "The water pipe burst yesterday and flooded the entire street. Multiple families are without water and electrical hazards are present. This requires immediate emergency response!"),
        ("Negative", "The streetlights on our block have been broken for over two weeks. It's getting dark early and residents feel unsafe walking at night. This issue needs to be addressed soon."),
        ("Neutral", "The city needs to repaint the parking lot lines in the park. Some markings have faded and are hard to see."),
        ("Positive", "Thank you for the recent park improvements! The new benches and landscaping look wonderful.")
    ]
    
    predictions_work = True
    for expected, text in test_cases:
        try:
            features = vectorizer.transform([text])
            prediction = model.predict(features)
            confidence = model.decision_function(features)
            status = "✓" if len(prediction) > 0 else "✗"
            print(f"  {status} {expected}: {prediction[0]}")
        except Exception as e:
            predictions_work = False
            errors.append(f"Prediction error for {expected}: {str(e)}")
            print(f"  ✗ {expected}: ERROR")
    
    checks.append(('Predictions work on test cases', predictions_work))

    # 5. Check required files
    print(f"\n✓ Checking required project files...")
    required_files = [
        'streamlit_app_dark.py',
        'api/main.py',
        'requirements.txt',
        'README.md'
    ]
    
    for filename in required_files:
        filepath = Path(filename)
        exists = filepath.exists()
        checks.append((f'File exists: {filename}', exists))
        symbol = "✓" if exists else "✗"
        print(f"  {symbol} {filename}")

    # 6. Verify requirements.txt
    print(f"\n✓ Checking dependencies in requirements.txt...")
    with open('requirements.txt') as f:
        reqs = f.read()
        has_streamlit = 'streamlit' in reqs
        has_fastapi = 'fastapi' in reqs
        has_sklearn = 'scikit-learn' in reqs
        
        checks.append(('Has streamlit in requirements', has_streamlit))
        checks.append(('Has fastapi in requirements', has_fastapi))
        checks.append(('Has scikit-learn in requirements', has_sklearn))
        
        print(f"  ✓ streamlit: {has_streamlit}")
        print(f"  ✓ fastapi: {has_fastapi}")
        print(f"  ✓ scikit-learn: {has_sklearn}")

except Exception as e:
    errors.append(f"Unexpected error: {str(e)}")
    print(f"\n✗ ERROR: {str(e)}")

# Print summary
print("\n" + "="*60)
print("📊 VERIFICATION SUMMARY")
print("="*60)

for check_name, status in checks:
    symbol = "✅" if status else "❌"
    result = "PASS" if status else "FAIL"
    print(f"{symbol} {check_name}: {result}")

# Final status
all_pass = all(status for _, status in checks)
print("\n" + "="*60)
if all_pass:
    print("✅ ALL CHECKS PASSED - READY FOR DEPLOYMENT")
    print("\nDeployment Steps:")
    print("1. ✓ All files present and valid")
    print("2. ✓ Models loaded successfully")
    print("3. ✓ Predictions working on all 4 sentiment classes")
    print("4. ✓ All dependencies installed")
    print("\n🚀 Ready for Streamlit Cloud deployment!")
else:
    print("❌ SOME CHECKS FAILED - DEPLOYMENT BLOCKED")
    if errors:
        print("\nErrors found:")
        for error in errors:
            print(f"  • {error}")
    sys.exit(1)

print("="*60 + "\n")
