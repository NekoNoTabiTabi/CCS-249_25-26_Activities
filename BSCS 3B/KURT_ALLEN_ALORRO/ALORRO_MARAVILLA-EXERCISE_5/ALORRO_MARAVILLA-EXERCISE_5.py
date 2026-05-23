"""
Train Skip-gram with Negative Sampling on a Wikipedia article,
then evaluate the embedding model with intrinsic tests and custom test sets.

Requirements:
    pip install requests beautifulsoup4 nltk gensim scikit-learn scipy
"""

import re
import math
import json
import random
from collections import Counter
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt

import requests
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA #addition for PCA visualization
import numpy as np


# 1. Use a Wikipedia article as your dataset (10 points)
# Updated to your chosen article: Artificial Intelligence
WIKI_URL = "https://en.wikipedia.org/wiki/Artificial_intelligence"
RANDOM_SEED = 42


def ensure_nltk():
    resources = ["punkt", "punkt_tab"]
    for r in resources:
        try:
            nltk.data.find(f"tokenizers/{r}")
        except LookupError:
            nltk.download(r)


def fetch_wikipedia_article(url: str) -> str:
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; SGNS-AI-Training/1.0)"
    }
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    content_div = soup.find("div", {"id": "mw-content-text"})
    if content_div is None:
        raise ValueError("Could not find Wikipedia article content.")

    paragraphs = content_div.find_all(["p", "li"])
    text_blocks = []

    for p in paragraphs:
        txt = p.get_text(" ", strip=True)
        if txt:
            text_blocks.append(txt)

    text = "\n".join(text_blocks)
    text = re.sub(r"\[[0-9]+\]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# 2. Preprocess the text coming from the selected corpus (10 points)
def preprocess_text(text: str) -> List[List[str]]:
    sentences = sent_tokenize(text)

    processed = []
    for sent in sentences:
        sent = sent.lower()
        sent = re.sub(r"[^a-z0-9\-\s]", " ", sent)
        sent = re.sub(r"\s+", " ", sent).strip()
        if not sent:
            continue

        tokens = word_tokenize(sent)

        cleaned = []
        for tok in tokens:
            tok = tok.strip("-")
            if not tok or tok.isdigit() or len(tok) < 2:
                continue
            cleaned.append(tok)

        if len(cleaned) >= 3:
            processed.append(cleaned)

    return processed


def corpus_stats(sentences: List[List[str]]) -> Dict[str, int]:
    flat = [w for s in sentences for w in s]
    vocab = set(flat)
    return {
        "num_sentences": len(sentences),
        "num_tokens": len(flat),
        "vocab_size": len(vocab),
    }


# 3. Train a Skip-gram with Negative Sampling model (10 points)
def train_sgns(sentences: List[List[str]]) -> Word2Vec:
    model = Word2Vec(
       sentences=sentences,
        vector_size=100, # What happens if we change this? Try 50, 200, 300 and see how it affects results.
        window=10,     #change to 10 for retraining
        min_count=1,
        workers=4,
        sg=1,          # 0 = CBOW, 1 = skip-gram
        negative=10,   # negative sampling
        epochs=200,
        sample=1e-3,
        alpha=0.025,
        min_alpha=0.0007,
        seed=RANDOM_SEED,
    )
    return model


def has_word(model: Word2Vec, word: str) -> bool:
    return word in model.wv.key_to_index


def cosine(model: Word2Vec, w1: str, w2: str) -> float:
    v1 = model.wv[w1].reshape(1, -1)
    v2 = model.wv[w2].reshape(1, -1)
    return float(cosine_similarity(v1, v2)[0][0])


def evaluate_relatedness(model: Word2Vec, test_pairs: List[Tuple[str, str, float]]):
    covered = []
    for w1, w2, score in test_pairs:
        if has_word(model, w1) and has_word(model, w2):
            sim = cosine(model, w1, w2)
            covered.append((w1, w2, score, sim))

    return {
        "covered_items": covered,
        "coverage": len(covered),
        "total": len(test_pairs),
    }


def evaluate_analogies(model: Word2Vec, analogies: List[Tuple[str, str, str, str]]):
    covered = 0
    correct = 0
    details = []

    for a, b, c, d in analogies:
        if all(has_word(model, w) for w in [a, b, c, d]):
            covered += 1
            try:
                preds = model.wv.most_similar(positive=[b, c], negative=[a], topn=5)
                predicted_words = [w for w, _ in preds]
                hit = d in predicted_words
                correct += int(hit)
                details.append({
                    "analogy": f"{a}:{b}::{c}:?",
                    "expected": d,
                    "predictions": predicted_words,
                    "correct_in_top5": hit
                })
            except KeyError:
                pass

    accuracy = correct / covered if covered else 0
    return {
        "coverage": covered,
        "total": len(analogies),
        "accuracy_top5": accuracy,
        "details": details
    }


def print_top_neighbors(model: Word2Vec, words: List[str], topn: int = 8):
    print("\n=== Nearest Neighbors ===")
    for word in words:
        if has_word(model, word):
            neighbors = model.wv.most_similar(word, topn=topn)
            print(f"\n{word}:")
            for neigh, score in neighbors:
                print(f"  {neigh:20s} {score:.4f}")
        else:
            print(f"\n{word}: [OOV]")


def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    ensure_nltk()

    print("Downloading Wikipedia article...")
    raw_text = fetch_wikipedia_article(WIKI_URL)

    print("Preprocessing text...")
    sentences = preprocess_text(raw_text)
    stats = corpus_stats(sentences)

    print("\n=== Corpus Stats ===")
    for k, v in stats.items():
        print(f"{k}: {v}")

    print("\nTraining Skip-gram with Negative Sampling...")
    model = train_sgns(sentences)

    print("\nVocabulary size learned:", len(model.wv))

    # 4. Evaluate the embeddings using a small test set (10 points)
    # Changed probe words to AI-relevant terms
    probe_words = [
        "intelligence", "machine", "learning", "neural", "networks",
        "turing", "algorithm", "data", "robot", "research"
    ]
    
    # 5. Report nearest neighbors, similarity scores, and test-set performance (10 points)
    print_top_neighbors(model, probe_words, topn=8)

    # Domain-specific relatedness test set (Updated)
    relatedness_test = [
        ("intelligence", "artificial", 0.95),
        ("machine", "learning", 0.90),
        ("neural", "networks", 0.95),
        ("deep", "learning", 0.85),
        ("turing", "test", 0.80),
        ("algorithm", "computation", 0.70),
        ("data", "information", 0.75),
        ("robot", "machine", 0.60),
        ("intelligence", "banana", 0.01), # Unrelated
        ("neural", "ocean", 0.01),       # Unrelated
    ]

    rel_results = evaluate_relatedness(model, relatedness_test)

    print("\n=== Relatedness Test Set ===")
    print(f"Coverage: {rel_results['coverage']}/{rel_results['total']}")
    for w1, w2, gold, pred in rel_results["covered_items"]:
        print(f"{w1:15s} - {w2:15s} | gold={gold:.2f} pred={pred:.4f}")

    # Small analogy-style test set (Updated)
    analogy_test = [
        ("human", "brain", "computer", "processor"),
        ("learning", "experience", "training", "data"),
        ("machine", "robot", "program", "software"),
    ]

    analogy_results = evaluate_analogies(model, analogy_test)

    print("\n=== Analogy Test Set ===")
    print(f"Coverage: {analogy_results['coverage']}/{analogy_results['total']}")
    print(f"Top-5 accuracy: {analogy_results['accuracy_top5']:.2%}")
    for item in analogy_results["details"]:
        print(json.dumps(item, ensure_ascii=False))

    # Direct Similarity Checks
    print("\n=== Direct Similarity Checks ===")
    check_pairs = [
        ("intelligence", "artificial"),
        ("machine", "learning"),
        ("neural", "networks"),
        ("intelligence", "robot"),
    ]
    for w1, w2 in check_pairs:
        if has_word(model, w1) and has_word(model, w2):
            print(f"{w1:15s} <-> {w2:15s}: {cosine(model, w1, w2):.4f}")



    # For retraining the Word2Vec Model Code Addition: Visualize the vectors using PCA 
    print("\n=== PCA Visualization ===")
    
    # 20 known words from the Artificial Intelligence corpus
    pca_words = [
        "intelligence", "artificial", "machine", "learning", "neural",
        "networks", "deep", "data", "information", "algorithm",
        "computation", "human", "brain", "computer", "processor",
        "research", "turing", "test", "robot", "software"
    ]

    # Filter words to ensure they are actually in the model's vocabulary
    valid_words = [w for w in pca_words if has_word(model, w)]
    
    if len(valid_words) > 0:
        # Extract the 100-dimensional vectors for these words
        word_vectors = np.array([model.wv[w] for w in valid_words])
        
        # Fit PCA to reduce dimensions from 100 to 2
        pca = PCA(n_components=2)
        vectors_2d = pca.fit_transform(word_vectors)
        
        #Scatter plot
        plt.figure(figsize=(12, 8))
        plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], edgecolors='k', c='skyblue', s=100)
        
        #annotate the points with their corresponding words
        for i, word in enumerate(valid_words):
            plt.annotate(word, xy=(vectors_2d[i, 0], vectors_2d[i, 1]), 
                         xytext=(5, 2), textcoords='offset points', 
                         ha='right', va='bottom', fontsize=12, fontweight='bold')
            
        plt.title('PCA Visualization of AI Word Embeddings (Window=10)', fontsize=16)
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.grid(True, linestyle='--', alpha=0.6)
        
        # This will pop open a window with your graph!
        plt.show() 
    else:
        print("None of the specified words were found in the vocabulary.")


    model.save("exercise_5_skipgram_sgns.model")
    print("\nSaved model to: exercise_5_skipgram_sgns.model")

    print("\nDone.")

if __name__ == "__main__":
    main()