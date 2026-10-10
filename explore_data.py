import pandas as pd
import os

print("=" * 70)
print("EXPLORING FAKE NEWS DATASET")
print("=" * 70)

# Load datasets
fake_df = pd.read_csv('data/Fake.csv')
true_df = pd.read_csv('data/True.csv')

print("\n📊 FAKE NEWS DATASET")
print(f"Total articles: {len(fake_df)}")
print(f"Columns: {list(fake_df.columns)}")
print(f"\nFirst article:")
print(f"Title: {fake_df['title'].iloc[0]}")
print(f"Text (first 200 chars): {fake_df['text'].iloc[0][:200]}")

print("\n📊 REAL NEWS DATASET")
print(f"Total articles: {len(true_df)}")
print(f"Columns: {list(true_df.columns)}")
print(f"\nFirst article:")
print(f"Title: {true_df['title'].iloc[0]}")
print(f"Text (first 200 chars): {true_df['text'].iloc[0][:200]}")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Total articles: {len(fake_df) + len(true_df)}")
print(f"Fake: {len(fake_df)} ({len(fake_df)/(len(fake_df)+len(true_df))*100:.1f}%)")
print(f"Real: {len(true_df)} ({len(true_df)/(len(fake_df)+len(true_df))*100:.1f}%)")

# Check for missing values
print("\n🔍 MISSING VALUES")
print(f"Fake news missing: {fake_df.isnull().sum().sum()}")
print(f"Real news missing: {true_df.isnull().sum().sum()}")