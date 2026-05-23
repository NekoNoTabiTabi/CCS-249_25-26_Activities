"""
Train Skip-gram with Negative Sampling on a Wikipedia article,
then evaluate the embedding model with intrinsic tests and custom test sets.

Requirements:
    pip install requests beautifulsoup4 nltk gensim scikit-learn scipy matplotlib

Optional:
    python -m nltk.downloader punkt stopwords
"""

import re
import json
import random
from typing import List, Tuple, Dict

import requests
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt


# Paste the Wikipedia link you want to use here. 
# The article should be reasonably long (at least a few thousand words) for good results.
WIKI_URL = "https://en.wikipedia.org/wiki/Climate_change"
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
        "User-Agent": "Mozilla/5.0 (compatible; SGNS-Gundam-Training/1.0)"
    }
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()

    # Extract main content text from the Wikipedia page
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
            if not tok:
                continue
            if tok.isdigit():
                continue
            if len(tok) < 2:
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


def train_sgns(sentences: List[List[str]], window: int = 5, vector_size: int = 100) -> Word2Vec:
    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,  # Try 50, 200, 300 and compare results.
        window=window,
        min_count=2,
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
    pred = []
    covered = []

    for w1, w2, score in test_pairs:
        if has_word(model, w1) and has_word(model, w2):
            sim = cosine(model, w1, w2)
            pred.append(sim)
            covered.append((w1, w2, score, sim))

    return {
        "covered_items": covered,
        "coverage": len(covered),
        "total": len(test_pairs),
        "avg_pred": float(np.mean(pred)) if pred else float("nan"),
    }


def evaluate_analogies(model: Word2Vec, analogies: List[Tuple[str, str, str, str]]):
    """
    Analogy format: a:b :: c:d
    Checks whether most_similar(positive=[b,c], negative=[a]) returns d.
    """
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

    accuracy = correct / covered if covered else float("nan")
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


def top_neighbors(model: Word2Vec, word: str, topn: int = 5):
    if has_word(model, word):
        return [(w, float(s)) for w, s in model.wv.most_similar(word, topn=topn)]
    return []


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

    print("\nTraining Skip-gram with Negative Sampling (window=5)...")
    model_w5 = train_sgns(sentences, window=5, vector_size=100)

    print("\nVocabulary size learned:", len(model_w5.wv))

    probe_words = [
        "climate", "warming", "emissions", "carbon", "energy",
        "renewable", "temperature", "atmosphere", "policy", "fossil"
    ]
    print_top_neighbors(model_w5, probe_words, topn=8)

    # Domain-specific relatedness test set
    # Higher score means should be more semantically related
    relatedness_test = [
        ("climate", "warming", 0.95),
        ("carbon", "emissions", 0.95),
        ("renewable", "energy", 0.90),
        ("fossil", "fuel", 0.90),
        ("sea", "level", 0.85),
        ("temperature", "warming", 0.85),
        ("policy", "agreement", 0.70),
        ("ice", "glacier", 0.80),
        ("climate", "banana", 0.05),
        ("ocean", "forest", 0.20),
        ("carbon", "banana", 0.02),
        ("renewable", "politics", 0.25),
    ]

    rel_results = evaluate_relatedness(model_w5, relatedness_test)

    print("\n=== Relatedness Test Set ===")
    print(f"Coverage: {rel_results['coverage']}/{rel_results['total']}")
    for w1, w2, gold, pred in rel_results["covered_items"]:
        print(f"{w1:10s} - {w2:10s} | gold={gold:.2f} pred={pred:.4f}")

    # Small analogy-style test set
    # These are intentionally tiny and corpus-dependent because a single article is a small dataset.
    # Change this based on what you find in the article and what words are present in the model.
    analogy_test = [
        ("emissions", "carbon", "temperature", "warming"),
        ("ocean", "sea", "glacier", "ice"),
        ("fossil", "fuel", "renewable", "energy"),
    ]

    analogy_results = evaluate_analogies(model_w5, analogy_test)

    print("\n=== Analogy Test Set ===")
    print(f"Coverage: {analogy_results['coverage']}/{analogy_results['total']}")
    print(f"Top-5 accuracy: {analogy_results['accuracy_top5']}")
    for item in analogy_results["details"]:
        print(json.dumps(item, ensure_ascii=False))

    # Example direct similarity checks
    print("\n=== Direct Similarity Checks ===")
    # Change these pairs based on what you expect to be related/unrelated in the article and what words are in the model.
    check_pairs = [
        ("climate", "warming"),
        ("carbon", "emissions"),
        ("renewable", "energy"),
        ("climate", "banana"),
    ]
    for w1, w2 in check_pairs:
        if has_word(model_w5, w1) and has_word(model_w5, w2):
            print(f"{w1:10s} <-> {w2:10s}: {cosine(model_w5, w1, w2):.4f}")
        else:
            print(f"{w1:10s} <-> {w2:10s}: OOV")

    print("\nRetraining with window size 10...")
    model_w10 = train_sgns(sentences, window=10, vector_size=100)
    rel_w10 = evaluate_relatedness(model_w10, relatedness_test)
    ana_w10 = evaluate_analogies(model_w10, analogy_test)

    print("\n=== Window Size Comparison ===")
    print(f"relatedness avg_pred (w=5):  {rel_results['avg_pred']:.4f}")
    print(f"relatedness avg_pred (w=10): {rel_w10['avg_pred']:.4f}")
    print(f"analogy top5 (w=5):  {analogy_results['accuracy_top5']}")
    print(f"analogy top5 (w=10): {ana_w10['accuracy_top5']}")

    print("\nGenerating PCA plot...")
    pca_words = [
        "climate", "warming", "carbon", "emissions", "energy", "renewable",
        "temperature", "atmosphere", "policy", "fossil", "fuel", "ocean",
        "sea", "level", "ice", "glacier", "forest", "drought", "storm", "mitigation"
    ]
    pca_words = [w for w in pca_words if has_word(model_w5, w)]
    X = np.array([model_w5.wv[w] for w in pca_words])
    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    coords = pca.fit_transform(X)

    plt.figure(figsize=(8, 6))
    plt.scatter(coords[:, 0], coords[:, 1])
    for word, (x, y) in zip(pca_words, coords):
        plt.text(x + 0.02, y + 0.02, word, fontsize=9)
    plt.title("PCA of Word Vectors (Climate Change)")
    plt.tight_layout()
    plt.savefig("unit5_pca.png", dpi=200)
    print("Saved PCA plot to: unit5_pca.png")

    # Save model
    model_w5.save("exercise_5_skipgram_sgns.model")
    print("\nSaved model to: exercise_5_skipgram_sgns.model")

    print("\nDone.")

main()