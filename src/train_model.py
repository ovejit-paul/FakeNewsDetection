import pandas as pd
import numpy as np
import string
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

nltk.download('punkt')
nltk.download('stopwords')

STOPWORDS = set(stopwords.words('english'))

def simple_preprocess(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t.isalpha() and t not in STOPWORDS]
    return " ".join(tokens)

def train(path_to_csv, out_path="../models/artifacts.pkl"):
    df = pd.read_csv(path_to_csv)
    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError("CSV must contain 'text' and 'label' columns.")
    
    df = df[['text', 'label']].dropna()
    df['label'] = df['label'].apply(lambda x: 1 if str(x).lower() in ['real', '1', 'true', 't'] else 0)
    df['text_clean'] = df['text'].astype(str).apply(simple_preprocess)

    X = df['text_clean'].values
    y = df['label'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    vec = TfidfVectorizer(ngram_range=(1,2), min_df=3, max_df=0.9)
    X_train_tfidf = vec.fit_transform(X_train)
    X_test_tfidf = vec.transform(X_test)

    clf = LogisticRegression(max_iter=1000, C=1.0, solver='liblinear')
    clf.fit(X_train_tfidf, y_train)

    preds = clf.predict(X_test_tfidf)
    acc = accuracy_score(y_test, preds)
    print("✅ Test accuracy:", acc)
    print(classification_report(y_test, preds, digits=4))

    artifacts = {"vectorizer": vec, "classifier": clf}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    joblib.dump(artifacts, out_path)
    print(f"✅ Saved model artifacts to {out_path}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Train fake/real news model")
    parser.add_argument('--data', required=True, help="Path to CSV with columns 'text' and 'label'")
    parser.add_argument('--out', default="../models/artifacts.pkl", help="Output artifact path")
    args = parser.parse_args()
    train(args.data, args.out)
