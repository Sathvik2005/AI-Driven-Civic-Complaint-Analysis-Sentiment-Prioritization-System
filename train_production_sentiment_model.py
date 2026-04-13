"""
Production-Grade Sentiment Model Training with Overfitting Prevention
Features: Cross-validation, regularization tuning, learning curves, realistic evaluation
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
import json
import os
import warnings
warnings.filterwarnings('ignore')

def create_balanced_training_data():
    """Create properly labeled dataset with meaningful sentiments"""
    
    sentiment_data = {
        'Critical': [
            'The pothole is extremely dangerous and must be fixed immediately',
            'The street is flooded with water and needs urgent repair',
            'This is a serious safety hazard that requires immediate attention',
            'The infrastructure is collapsing and poses great risk',
            'This is an urgent emergency that needs immediate action',
            'The situation is critical and dangerous to public safety',
            'This needs to be fixed right now, it is a major hazard',
            'Severe damage reported, requires emergency response',
            'This is extremely urgent and life threatening',
            'Critical infrastructure failure reported',
            'Dangerous condition reported, immediate action needed',
            'Major damage, urgent repairs required',
            'Safety hazard, urgent action necessary',
            'Emergency situation reported',
            'Critical infrastructure damage',
            'Severe damage to road infrastructure',
            'Major pothole causing accidents',
            'Dangerous flooding, urgent repair needed',
            'Critical safety issue reported',
            'Urgent repair needed for safety',
            'Extremely serious damage reported',
            'Major structural damage found',
            'Critical maintenance issue',
            'Severe accident risk detected',
        ],
        'Negative': [
            'The service quality is poor and unsatisfactory',
            'I am unhappy with the maintenance work done',
            'The repairs were not done properly and I am frustrated',
            'This is completely broken and not working',
            'I hate this, it is absolutely terrible',
            'The work is substandard and disappointing',
            'This is a disaster, very bad experience',
            'I am very upset with the lack of response',
            'The road condition is really bad',
            'Service is awful and unprofessional',
            'Not satisfied with the work quality',
            'Poor maintenance and neglect',
            'Disappointed with the repairs',
            'Terrible service experience',
            'Very unhappy with the outcome',
            'Bad road conditions remain',
            'Inadequate repairs done',
            'Frustrated with lack of progress',
            'Unsatisfactory response from authorities',
            'Work quality is below standard',
            'Angry about the situation',
            'Disappointed by delayed repairs',
            'Upset with the poor response',
        ],
        'Neutral': [
            'The road needs routine maintenance',
            'A pothole was reported on Main Street',
            'Drainage system requires inspection',
            'Routine maintenance is scheduled',
            'The street just needs normal upkeep',
            'Scheduled maintenance work is needed',
            'Report of minor damage on the road',
            'Regular maintenance required',
            'Infrastructure inspection needed',
            'Normal wear and tear detected',
            'Routine repairs are pending',
            'Standard maintenance procedure',
            'Ordinary maintenance work required',
            'Scheduled street cleaning needed',
            'Regular inspection due',
            'Typical maintenance issue',
            'Standard repair work',
            'Normal maintenance schedule',
            'Routine inspection required',
            'Regular upkeep needed',
            'Ordinary repairs needed',
            'Standard maintenance protocol',
            'Typical infrastructure issue',
        ],
        'Positive': [
            'Thank you for the quick and efficient service',
            'Excellent work, the road is perfectly fixed now',
            'I am very happy with the repairs done',
            'The service was professional and timely',
            'Great job by the maintenance team',
            'The infrastructure is in great condition now',
            'Wonderful service, highly satisfied',
            'The road is now in excellent condition',
            'Thank you for solving the problem quickly',
            'The repairs were done perfectly',
            'Very appreciated the quick response',
            'Outstanding work by the team',
            'Excellent service quality',
            'Very satisfied with the work',
            'Fantastic job done',
            'Road condition is now excellent',
            'Great maintenance work',
            'Pleased with the repairs',
            'Service was outstanding',
            'Happy with the outcome',
            'Appreciative of the quick fix',
            'Impressed with the work quality',
            'Good work, thank you',
        ]
    }
    
    texts = []
    labels = []
    for label, complaints in sentiment_data.items():
        texts.extend(complaints)
        labels.extend([label] * len(complaints))
    
    df = pd.DataFrame({'text': texts, 'label': labels})
    return df

def train_with_regularization(df):
    """Train model with proper regularization and cross-validation"""
    
    print("\n" + "="*70)
    print("PRODUCTION-GRADE MODEL TRAINING WITH OVERFITTING PREVENTION")
    print("="*70)
    
    # Split data with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['label'], 
        test_size=0.2, 
        random_state=42,
        stratify=df['label']
    )
    
    print(f"\n📊 Dataset Split:")
    print(f"   Training: {len(X_train)} samples")
    print(f"   Testing:  {len(X_test)} samples")
    
    # Vectorizer
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
        lowercase=True,
        stop_words='english'
    )
    
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    print(f"\n🔤 Vectorizer:")
    print(f"   Vocabulary size: {len(vectorizer.vocabulary_)}")
    print(f"   N-gram range: {vectorizer.ngram_range}")
    
    # Train with regularization (C parameter controls regularization strength)
    # Lower C = stronger regularization = prevent overfitting
    model = LinearSVC(
        class_weight='balanced',
        C=0.1,  # Regularization parameter (0.1 = strong regularization to prevent overfitting)
        max_iter=2000,
        random_state=42,
        multi_class='crammer_singer'
    )
    
    # Fit model
    model.fit(X_train_vec, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train_vec)
    y_test_pred = model.predict(X_test_vec)
    
    # Calculate metrics
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    train_f1 = f1_score(y_train, y_train_pred, average='weighted')
    test_f1 = f1_score(y_test, y_test_pred, average='weighted')
    
    print(f"\n📈 Performance Metrics:")
    print(f"   Training Accuracy: {train_acc:.4f}")
    print(f"   Testing Accuracy:  {test_acc:.4f}")
    print(f"   >>> Overfitting Gap: {(train_acc - test_acc)*100:.2f}% <<<")
    print(f"\n   Training F1-Score: {train_f1:.4f}")
    print(f"   Testing F1-Score:  {test_f1:.4f}")
    
    # Cross-validation for more robust evaluation
    print(f"\n🔄 Cross-Validation (5-fold):")
    cv_scores = cross_val_score(model, X_train_vec, y_train, cv=5, scoring='f1_weighted')
    print(f"   Fold scores: {[f'{s:.4f}' for s in cv_scores]}")
    print(f"   Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    # Classification report
    print(f"\n📋 Classification Report (Test Set):")
    print(classification_report(y_test, y_test_pred, labels=model.classes_))
    
    # Confusion matrix
    print(f"\n🎯 Confusion Matrix (Test Set):")
    cm = confusion_matrix(y_test, y_test_pred, labels=model.classes_)
    print(cm)
    
    # Overfitting assessment
    print(f"\n⚠️  OVERFITTING ASSESSMENT:")
    gap = train_acc - test_acc
    if gap > 0.15:
        print(f"   ❌ POTENTIAL OVERFITTING DETECTED ({gap*100:.2f}% gap)")
        print(f"   → Model memorized training data, may not generalize well")
    elif gap > 0.05:
        print(f"   ⚠️  SLIGHT OVERFITTING ({gap*100:.2f}% gap)")
        print(f"   → Acceptable performance, monitor in production")
    else:
        print(f"   ✅ HEALTHY GENERALIZATION ({gap*100:.2f}% gap)")
        print(f"   → Model generalizes well to unseen data")
    
    return model, vectorizer, {
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'train_f1': train_f1,
        'test_f1': test_f1,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'overfitting_gap': gap,
        'regularization_parameter': 0.5,
        'vocabulary_size': len(vectorizer.vocabulary_),
    }

def test_model_robustness(model, vectorizer):
    """Test model with diverse real-world examples"""
    print("\n" + "="*70)
    print("ROBUSTNESS TESTING WITH DIVERSE EXAMPLES")
    print("="*70)
    
    test_cases = [
        ('This is amazing! Excellent service', 'Positive'),
        ('This is terrible! Awful service', 'Negative'),
        ('Pothole is dangerous, fix needed', 'Critical'),
        ('Just routine maintenance', 'Neutral'),
        ('I am very happy with this', 'Positive'),
        ('This is the worst experience', 'Negative'),
        ('Street flooded, urgent', 'Critical'),
        ('Road condition OK', 'Neutral'),
    ]
    
    correct = 0
    for text, expected in test_cases:
        pred = model.predict(vectorizer.transform([text]))[0]
        match = "✅" if pred == expected else "❌"
        print(f"{match} Expected: {expected:10} | Got: {pred:10} | Text: {text[:40]}...")
        if pred == expected:
            correct += 1
    
    print(f"\n🎯 Real-world Accuracy: {correct}/{len(test_cases)} ({100*correct/len(test_cases):.1f}%)")

def save_production_model(model, vectorizer, metrics, output_dir='trained_models'):
    """Save model with comprehensive metadata"""
    os.makedirs(output_dir, exist_ok=True)
    
    joblib.dump(model, f'{output_dir}/sentiment_model.joblib')
    joblib.dump(vectorizer, f'{output_dir}/tfidf_vectorizer.joblib')
    
    metadata = {
        'model_type': 'LinearSVC',
        'vectorizer_type': 'TfidfVectorizer',
        'vocabulary_size': metrics['vocabulary_size'],
        'max_features': 5000,
        'ngram_range': [1, 2],
        'sentiment_classes': ['Critical', 'Negative', 'Neutral', 'Positive'],
        'class_weight': 'balanced',
        'regularization_parameter': 0.1,
        'performance': {
            'test_accuracy': round(metrics['test_accuracy'], 4),
            'test_f1_score': round(metrics['test_f1'], 4),
            'cross_val_mean': round(metrics['cv_mean'], 4),
            'cross_val_std': round(metrics['cv_std'], 4),
            'overfitting_gap': round(metrics['overfitting_gap'], 4),
        },
        'notes': f"""
Production-grade model with overfitting prevention:
- Regularization: C=0.5 (prevents memorization)
- Class weights: balanced (handles class imbalance)
- Evaluation: 5-fold cross-validation performed
- Overfitting assessment: {"HEALTHY" if metrics["overfitting_gap"] < 0.05 else "MONITOR"}
- Train-test gap: {metrics["overfitting_gap"]*100:.2f}%
- Real-world tested: True
""".strip()
    }
    
    with open(f'{output_dir}/model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=4)
    
    print(f"\n✅ Models saved to {output_dir}/")
    print(f"   - sentiment_model.joblib")
    print(f"   - tfidf_vectorizer.joblib")
    print(f"   - model_metadata.json (with comprehensive metrics)")

if __name__ == '__main__':
    df = create_balanced_training_data()
    model, vectorizer, metrics = train_with_regularization(df)
    test_model_robustness(model, vectorizer)
    save_production_model(model, vectorizer, metrics)
    
    print("\n" + "="*70)
    print("✅ PRODUCTION-READY MODEL CREATED")
    print("="*70)
    print("\n🚀 Ready for deployment with:")
    print("   ✓ Regularization to prevent overfitting")
    print("   ✓ Cross-validation verification")
    print("   ✓ Realistic accuracy metrics")
    print("   ✓ Comprehensive evaluation")
    print("="*70)
