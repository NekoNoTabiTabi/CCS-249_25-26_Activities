"""
Train Skip-gram with Negative Sampling on a Wikipedia article,
then evaluate the embedding model with intrinsic tests and custom test sets.

Requirements:
    pip install requests beautifulsoup4 nltk gensim scikit-learn scipy

Optional:
    python -m nltk.downloader punkt stopwords
"""

import re
import math
import json
import random
from collections import Counter
from typing import List, Tuple, Dict

import requests
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np


# Paste the Wikipedia link you want to use here. 
# The article should be reasonably long (at least a few thousand words) for good results.
WIKI_URL = "https://en.wikipedia.org/wiki/Masamune"
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


def train_sgns(sentences: List[List[str]], window_size: int = 5) -> Word2Vec:
    model = Word2Vec(
        sentences=sentences,
        vector_size=100,
        window=window_size,
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
    gold = []
    pred = []
    covered = []

    for w1, w2, score in test_pairs:
        if has_word(model, w1) and has_word(model, w2):
            sim = cosine(model, w1, w2)
            gold.append(score)
            pred.append(sim)
            covered.append((w1, w2, score, sim))

    return {
        "covered_items": covered,
        "coverage": len(covered),
        "total": len(test_pairs),
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


def summarize_scores(model: Word2Vec, relatedness_test, analogy_test, check_pairs):
    rel_results = evaluate_relatedness(model, relatedness_test)
    analogy_results = evaluate_analogies(model, analogy_test)

    pair_scores = []
    for w1, w2 in check_pairs:
        if has_word(model, w1) and has_word(model, w2):
            pair_scores.append((w1, w2, cosine(model, w1, w2)))
        else:
            pair_scores.append((w1, w2, None))

    avg_rel = (
        sum(pred for _, _, _, pred in rel_results["covered_items"]) / rel_results["coverage"]
        if rel_results["coverage"] else float("nan")
    )

    return {
        "rel_coverage": f"{rel_results['coverage']}/{rel_results['total']}",
        "rel_avg_pred": avg_rel,
        "analogy_coverage": f"{analogy_results['coverage']}/{analogy_results['total']}",
        "analogy_top5": analogy_results["accuracy_top5"],
        "pair_scores": pair_scores,
    }


def plot_pca_words(
    model: Word2Vec,
    candidate_words: List[str],
    min_words: int = 20,
    out_path: str = "pca_20_words.png",
):
    known = [w for w in candidate_words if has_word(model, w)]

    # Backfill from vocabulary if fewer than 20 known words
    if len(known) < min_words:
        for w in model.wv.index_to_key:
            if w not in known:
                known.append(w)
            if len(known) >= min_words:
                break

    known = known[:max(min_words, len(known))]
    vectors = np.array([model.wv[w] for w in known])

    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    coords = pca.fit_transform(vectors)

    plt.figure(figsize=(12, 8))
    plt.scatter(coords[:, 0], coords[:, 1], s=35)

    for i, word in enumerate(known):
        plt.annotate(word, (coords[i, 0], coords[i, 1]), fontsize=9)

    evr = pca.explained_variance_ratio_
    plt.title(f"PCA of Word2Vec embeddings ({len(known)} words)")
    plt.xlabel(f"PC1 ({evr[0]*100:.2f}% var)")
    plt.ylabel(f"PC2 ({evr[1]*100:.2f}% var)")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.show()

    return known, evr


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

    min_tokens = 3000
    print(f"Token count: {stats['num_tokens']}")
    if stats["num_tokens"] < min_tokens:
        print("WARNING: Article may be too short. Pick a longer Wikipedia page.")
    else:
        print("OK: Article length is sufficient.")

    probe_words = [
        "masamune", "sword", "japanese", "smith", "blade",
        "katana", "kamakura", "period", "steel", "legend"
    ]

    relatedness_test = [
        ("masamune", "sword", 0.95),
        ("masamune", "smith", 0.90),
        ("sword", "blade", 0.92),
        ("katana", "sword", 0.93),
        ("japan", "japanese", 0.88),
        ("kamakura", "period", 0.82),
        ("masamune", "kitchen", 0.08),
        ("sword", "office", 0.03),
    ]

    analogy_test = [
        ("japan", "japanese", "sword", "katana"),
        ("smith", "blade", "sword", "katana"),
        ("kamakura", "period", "heian", "period"),
    ]

    check_pairs = [
        ("masamune", "sword"),
        ("masamune", "smith"),
        ("katana", "blade"),
        ("masamune", "office"),
    ]

    # OLD model (window=5)
    print("\nTraining OLD model (window=5)...")
    old_model = train_sgns(sentences, window_size=5)
    old_scores = summarize_scores(old_model, relatedness_test, analogy_test, check_pairs)

    # NEW model (window=10)
    print("\nTraining NEW model (window=10)...")
    new_model = train_sgns(sentences, window_size=10)
    new_scores = summarize_scores(new_model, relatedness_test, analogy_test, check_pairs)

    print("\n=== Comparison (OLD vs NEW) ===")
    print(f"OLD window=5 | relatedness coverage: {old_scores['rel_coverage']} | avg predicted sim: {old_scores['rel_avg_pred']:.4f} | analogy top-5: {old_scores['analogy_top5']}")
    print(f"NEW window=10 | relatedness coverage: {new_scores['rel_coverage']} | avg predicted sim: {new_scores['rel_avg_pred']:.4f} | analogy top-5: {new_scores['analogy_top5']}")

    print("\nPairwise similarity comparison:")
    print("word1       word2       OLD        NEW")
    for (w1, w2, old_s), (_, _, new_s) in zip(old_scores["pair_scores"], new_scores["pair_scores"]):
        old_txt = f"{old_s:.4f}" if old_s is not None else "OOV"
        new_txt = f"{new_s:.4f}" if new_s is not None else "OOV"
        print(f"{w1:10s}  {w2:10s}  {old_txt:8s}  {new_txt:8s}")

    # PCA visualization (at least 20 known words)
    pca_candidates = [
        "masamune", "sword", "smith", "blade", "katana", "japanese", "japan",
        "kamakura", "period", "legend", "steel", "forged", "weapon", "samurai",
        "history", "craft", "tradition", "famous", "master", "school", "style",
        "edge", "tanto", "wakizashi", "shogun"
    ]
    words_used, evr = plot_pca_words(
        new_model,
        pca_candidates,
        min_words=20,
        out_path="pca_20_words.png"
    )
    print(f"\nPCA plot saved to: pca_20_words.png")
    print(f"Words plotted ({len(words_used)}): {words_used[:20]}{' ...' if len(words_used) > 20 else ''}")
    print(f"Explained variance: PC1={evr[0]:.4f}, PC2={evr[1]:.4f}")

    # Keep saving NEW model per task
    new_model.save("exercise_5_skipgram_sgns_window10.model")
    print("\nSaved model to: exercise_5_skipgram_sgns_window10.model")
    print("\nDone.")

main()