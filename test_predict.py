import sys
sys.path.insert(0, '.')
from src.preprocess import clean_text
import joblib

model = joblib.load('models/fake_news_model.pkl')
vectorizer = joblib.load('models/tfidf_vectorizer.pkl')

tests = [
    "BREAKING: Scientists confirm drinking bleach cures all diseases. The mainstream media doesn't want you to know this secret!",
    "Federal Reserve raises interest rates by 0.25 percent to combat inflation according to official statement.",
    "Government hiding alien technology in secret underground base, whistleblowers reveal deep state conspiracy.",
]

for text in tests:
    cleaned = clean_text(text)
    tfidf = vectorizer.transform([cleaned])
    proba = model.predict_proba(tfidf)[0]
    label = 'FAKE' if proba[1] > 0.5 else 'REAL'
    print(f"{label} ({max(proba)*100:.1f}%) - {text[:60]}...")
