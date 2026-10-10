import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)

# Run preprocessing if not already done
from src.preprocess import load_and_prepare_data


def train():
    print("=" * 50)
    print("TRAINING FAKE NEWS DETECTOR MODEL")
    print("=" * 50)

    # Load or create processed data
    if os.path.exists('data/processed_data.csv'):
        print("Loading preprocessed data...")
        df = pd.read_csv('data/processed_data.csv')
    else:
        print("Preprocessed data not found. Running preprocessing...")
        df = load_and_prepare_data()

    X = df['cleaned_content']
    y = df['label']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTraining set: {len(X_train)} articles")
    print(f"Test set:     {len(X_test)} articles")

    # TF-IDF Vectorization
    print("\nVectorizing text with TF-IDF...")
    vectorizer = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),   # unigrams + bigrams
        min_df=2,
        max_df=0.95
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf  = vectorizer.transform(X_test)

    # Train Logistic Regression
    print("Training Logistic Regression model...")
    model = LogisticRegression(
        max_iter=1000,
        C=1.0,
        solver='lbfgs',
        n_jobs=-1
    )
    model.fit(X_train_tfidf, y_train)

    # Evaluate
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)

    print("\n" + "=" * 50)
    print("MODEL PERFORMANCE")
    print("=" * 50)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['REAL', 'FAKE']))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save model and vectorizer
    os.makedirs('models', exist_ok=True)
    joblib.dump(model,      'models/fake_news_model.pkl')
    joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')

    print("\nModel saved:      models/fake_news_model.pkl")
    print("Vectorizer saved: models/tfidf_vectorizer.pkl")

    # Save top FAKE-indicating words for explainability
    feature_names = vectorizer.get_feature_names_out()
    coefficients  = model.coef_[0]
    top_fake_indices = np.argsort(coefficients)[-30:][::-1]
    top_real_indices = np.argsort(coefficients)[:30]

    top_fake_words = [feature_names[i] for i in top_fake_indices]
    top_real_words = [feature_names[i] for i in top_real_indices]

    print("\nTop words indicating FAKE news:")
    print(", ".join(top_fake_words[:15]))
    print("\nTop words indicating REAL news:")
    print(", ".join(top_real_words[:15]))

    return model, vectorizer, accuracy


if __name__ == "__main__":
    train()
