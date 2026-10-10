import joblib
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from preprocess import clean_text


def load_model():
    """Load the saved model and vectorizer."""
    model      = joblib.load('models/fake_news_model.pkl')
    vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
    return model, vectorizer


def predict(text, model=None, vectorizer=None):
    """
    Predict whether a news article is FAKE or REAL.
    Returns a dict with label, confidence, and probability scores.
    """
    if model is None or vectorizer is None:
        model, vectorizer = load_model()

    cleaned = clean_text(text)
    tfidf   = vectorizer.transform([cleaned])
    proba   = model.predict_proba(tfidf)[0]

    fake_prob = proba[1]   # index 1 = FAKE
    real_prob = proba[0]   # index 0 = REAL

    label      = "FAKE" if fake_prob > 0.5 else "REAL"
    confidence = max(fake_prob, real_prob) * 100

    return {
        "label":      label,
        "confidence": round(confidence, 2),
        "fake_prob":  round(fake_prob * 100, 2),
        "real_prob":  round(real_prob * 100, 2),
    }


if __name__ == "__main__":
    print("=" * 50)
    print("FAKE NEWS DETECTOR — TEST PREDICTIONS")
    print("=" * 50)

    test_articles = [
        {
            "title": "BREAKING: Scientists confirm drinking bleach cures all diseases",
            "text":  "Shocking new study reveals that household bleach, when consumed in small amounts, can cure cancer, diabetes, and COVID-19. The mainstream media doesn't want you to know this secret!"
        },
        {
            "title": "Federal Reserve raises interest rates by 0.25 percent",
            "text":  "The Federal Reserve announced Wednesday it would raise its benchmark interest rate by a quarter percentage point, continuing its efforts to bring down inflation that has remained above its 2 percent target."
        },
        {
            "title": "Government hiding alien technology in secret underground base",
            "text":  "Whistleblowers reveal the deep state has been concealing advanced extraterrestrial technology for decades. The truth is being suppressed by globalist elites who control the mainstream media narrative."
        },
    ]

    model, vectorizer = load_model()

    for i, article in enumerate(test_articles, 1):
        result = predict(article['title'] + ' ' + article['text'], model, vectorizer)
        print(f"\nArticle {i}: {article['title'][:60]}...")
        print(f"  Prediction : {result['label']}")
        print(f"  Confidence : {result['confidence']}%")
        print(f"  FAKE prob  : {result['fake_prob']}%")
        print(f"  REAL prob  : {result['real_prob']}%")
