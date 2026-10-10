import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import os

# Download required NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    """Clean and preprocess a single text string."""
    if not isinstance(text, str):
        return ""

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)

    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Tokenize, remove stopwords, stem
    tokens = text.split()
    tokens = [stemmer.stem(w) for w in tokens if w not in stop_words and len(w) > 2]

    return ' '.join(tokens)


def load_and_prepare_data():
    """Load Fake.csv and True.csv, label them, and combine."""
    print("Loading datasets...")

    fake_df = pd.read_csv('data/Fake.csv')
    true_df = pd.read_csv('data/True.csv')

    # Add labels: 1 = FAKE, 0 = REAL
    fake_df['label'] = 1
    true_df['label'] = 0

    print(f"  Fake news articles: {len(fake_df)}")
    print(f"  Real news articles: {len(true_df)}")

    # Combine datasets
    df = pd.concat([fake_df, true_df], ignore_index=True)

    # Combine title + text for richer features
    df['content'] = df['title'].fillna('') + ' ' + df['text'].fillna('')

    print(f"  Total articles: {len(df)}")
    print("Cleaning text (this may take a minute)...")

    df['cleaned_content'] = df['content'].apply(clean_text)

    # Drop rows with empty cleaned content
    df = df[df['cleaned_content'].str.len() > 10].reset_index(drop=True)

    print(f"  Articles after cleaning: {len(df)}")

    # Save processed data
    os.makedirs('data', exist_ok=True)
    df[['cleaned_content', 'label', 'title']].to_csv('data/processed_data.csv', index=False)
    print("Saved: data/processed_data.csv")

    return df


if __name__ == "__main__":
    print("=" * 50)
    print("PREPROCESSING DATA")
    print("=" * 50)
    df = load_and_prepare_data()
    print("\nSample cleaned text:")
    print(df['cleaned_content'].iloc[0][:200])
    print("\nLabel distribution:")
    print(df['label'].value_counts().rename({1: 'FAKE', 0: 'REAL'}))
    print("\nPreprocessing complete!")
