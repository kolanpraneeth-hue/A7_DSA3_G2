#!/usr/bin/env python3
"""
Simple TF-IDF + cosine similarity baseline (original lexical method).
Useful for comparing against the semantic engine.
"""

from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from preprocess import extract_text, clean_text


def tfidf_similarity(query_path: str, ref_path: str) -> float:
    q = clean_text(extract_text(query_path))
    r = clean_text(extract_text(ref_path))

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform([q, r])
    score = cosine_similarity(matrix[0:1], matrix[1:2])[0][0]
    return round(float(score) * 100, 1)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("-q", "--query", required=True)
    p.add_argument("-r", "--ref", required=True)
    args = p.parse_args()

    pct = tfidf_similarity(args.query, args.ref)
    print(f"TF-IDF Cosine Similarity: {pct}%")
