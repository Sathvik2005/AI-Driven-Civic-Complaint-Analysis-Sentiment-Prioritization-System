"""
Improved Sentiment Model Training with Balanced Class Weights
Addresses: Class imbalance, proper word-meaning based classification
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import json
import os

def create_balanced_training_data():
    """Create a properly labeled dataset with meaningful sentiments based on word meanings"""
    
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
    
    # Create DataFrame
    texts = []
    labels = []
    for label, complaints in sentiment_data.items():
        texts.extend(complaints)
        labels.extend([label] * len(complaints))
    
    df = pd.DataFrame({'text': texts, 'label': labels})
    print(f"Dataset created: {len(df)} samples")
    print("\nClass Distribution:")
    print(df['label'].value_counts().sort_index())
    
    return df

def train_improved_model(df):
    """Train model with balanced class weights for better multi-class performance"""
    
    print("\n" + "="*60)
    print("Training Improved Sentiment Model")
    print("="*60)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['label'], 
        test_size=0.2, 
        random_state=42,
        stratify=df['label']
    )
    
    # Vectorizer with better parameters
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
    
    # Train LinearSVC with balanced class weights
    model = LinearSVC(
        class_weight='balanced',  # CRITICAL: Handles class imbalance
        max_iter=2000,
        random_state=42,
        C=1.0,
        multi_class='crammer_singer'
    )
    
    model.fit(X_train_vec, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_vec)
    accuracy = (y_pred == y_test).mean()
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"\nTest Accuracy: {accuracy:.4f}")
    print(f"Weighted F1-Score: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    print(cm)
    
    return model, vectorizer

def test_model(model, vectorizer):
    """Test with real-world examples"""
    print("\n" + "="*60)
    print("Testing with Real-World Examples")
    print("="*60)
    
    test_texts = [
        'This is amazing! The service is excellent and I am very happy',
        'This is terrible! The service is awful and I am very upset',
        'The street is flooded with water and needs repair',
        'Everything is okay, just routine maintenance needed',
        'The pothole is extremely dangerous and must be fixed immediately',
        'Thank you for the quick service, much appreciated',
        'I hate this, it is completely broken and useless',
        'The road is damaged and broken, needs immediate attention',
        'Good work, thank you for helping us',
        'This is the worst experience I have ever had',
    ]
    
    for text in test_texts:
        features = vectorizer.transform([text])
        pred = model.predict(features)[0]
        decision = model.decision_function(features)[0]
        print(f'\nText: "{text[:60]}..."')
        print(f'→ Prediction: {pred}')
        print(f'  Confidence scores: {dict(zip(model.classes_, decision))}')

def save_models(model, vectorizer, output_dir='trained_models'):
    """Save the trained models and metadata"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save models
    joblib.dump(model, f'{output_dir}/sentiment_model.joblib')
    joblib.dump(vectorizer, f'{output_dir}/tfidf_vectorizer.joblib')
    
    # Save metadata
    metadata = {
        'model_type': 'LinearSVC',
        'vectorizer_type': 'TfidfVectorizer',
        'vocabulary_size': len(vectorizer.vocabulary_),
        'max_features': vectorizer.max_features,
        'ngram_range': vectorizer.ngram_range,
        'sentiment_classes': list(model.classes_),
        'class_weight': 'balanced',
        'notes': 'Improved model with balanced class weights for better multi-class performance'
    }
    
    with open(f'{output_dir}/model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=4)
    
    print(f"\n✅ Models saved to {output_dir}/")

if __name__ == '__main__':
    # Create balanced dataset
    df = create_balanced_training_data()
    
    # Train improved model
    model, vectorizer = train_improved_model(df)
    
    # Test with real examples
    test_model(model, vectorizer)
    
    # Save models
    save_models(model, vectorizer)
    
    print("\n" + "="*60)
    print("✅ Training Complete! Model is ready for deployment")
    print("="*60)
