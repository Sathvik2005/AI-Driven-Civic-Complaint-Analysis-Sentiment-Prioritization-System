"""Test sentiment model to verify real-world classification"""
import joblib
import sys

# Load models
print("Loading models...")
model = joblib.load('trained_models/sentiment_model.joblib')
vectorizer = joblib.load('trained_models/tfidf_vectorizer.joblib')

# Test with diverse examples
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

print("\n" + "="*80)
print("SENTIMENT MODEL TESTING")
print("="*80)

for text in test_texts:
    features = vectorizer.transform([text])
    prediction = model.predict(features)[0]
    proba = model.decision_function(features)[0]
    print(f'\nText: "{text}"')
    print(f'Prediction: {prediction}')
    print(f'Decision Score: {proba}')
    print('-'*80)

print("\nModel classes:", model.classes_)
print("Model type:", type(model))
print("Vectorizer type:", type(vectorizer))
