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

WIKI_URL = "https://en.wikipedia.org/wiki/Algorithmic_radicalization"
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
        "User-Agent": "Mozilla/5.0 (compatible; SGNS-AlgorithmicRadicalization-Training/1.0)"
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


def train_sgns(sentences: List[List[str]], window: int = 5) -> Word2Vec:
    model = Word2Vec(
        sentences=sentences,
        vector_size=100,
        window=window,
        min_count=1,
        workers=4,
        sg=1,
        negative=10,
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


def plot_pca(models: List[Tuple[Word2Vec, int]], words: List[str]):
    fig, axes = plt.subplots(1, len(models), figsize=(20, 8))
    if len(models) == 1:
        axes = [axes]
    for ax, (model, window) in zip(axes, models):
        vocab_words = [w for w in words if has_word(model, w)]
        vectors = np.array([model.wv[w] for w in vocab_words])
        pca = PCA(n_components=2, random_state=RANDOM_SEED)
        coords = pca.fit_transform(vectors)
        ax.scatter(coords[:, 0], coords[:, 1], s=60, color="steelblue", zorder=2)
        for i, word in enumerate(vocab_words):
            ax.annotate(word, (coords[i, 0], coords[i, 1]),
                        textcoords="offset points", xytext=(6, 4), fontsize=9)
        ax.set_title(f"PCA — Algorithmic Radicalization (window={window})", fontsize=12)
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
        ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig("pca_both_models.png", dpi=150)
    plt.show()
    print("Saved to: pca_both_models.png")


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

    probe_words = [
        "radicalization", "algorithm", "recommendation", "extremism", "platform",
        "content", "social", "media", "online", "users"
    ]

    relatedness_test = [
        ("radicalization", "extremism",      0.90),
        ("algorithm",      "recommendation", 0.85),
        ("echo",           "chamber",        0.95),
        ("filter",         "bubble",         0.90),
        ("platform",       "social",         0.75),
        ("content",        "media",          0.75),
        ("hate",           "speech",         0.85),
        ("online",         "internet",       0.80),
        ("radicalization", "kitchen",        0.02),
        ("algorithm",      "flower",         0.02),
        ("extremism",      "users",          0.35),
        ("conspiracy",     "theory",         0.90),
    ]

    analogy_test = [
        ("online",         "internet",    "social",         "media"),
        ("radicalization", "extremism",   "recommendation", "algorithm"),
        ("filter",         "bubble",      "echo",           "chamber"),
        ("platform",       "youtube",     "users",          "viewers"),
    ]

    check_pairs = [
        ("radicalization", "extremism"),
        ("algorithm",      "recommendation"),
        ("echo",           "chamber"),
        ("radicalization", "kitchen"),
    ]

    trained_models = []

    for window in [5, 10]:
        print(f"\n{'='*45}")
        print(f"  Training Skip-gram with Negative Sampling (window={window})...")
        print(f"{'='*45}")

        model = train_sgns(sentences, window=window)
        model.save(f"exercise_5_skipgram_w{window}.model")
        print(f"Vocabulary size learned: {len(model.wv)}")

        print_top_neighbors(model, probe_words, topn=8)

        rel_results = evaluate_relatedness(model, relatedness_test)
        print("\n=== Relatedness Test Set ===")
        print(f"Coverage: {rel_results['coverage']}/{rel_results['total']}")
        for w1, w2, gold, pred in rel_results["covered_items"]:
            print(f"{w1:15s} - {w2:15s} | gold={gold:.2f}  pred={pred:.4f}")

        analogy_results = evaluate_analogies(model, analogy_test)
        print("\n=== Analogy Test Set ===")
        print(f"Coverage: {analogy_results['coverage']}/{analogy_results['total']}")
        print(f"Top-5 accuracy: {analogy_results['accuracy_top5']}")
        for item in analogy_results["details"]:
            print(json.dumps(item, ensure_ascii=False))

        print("\n=== Direct Similarity Checks ===")
        for w1, w2 in check_pairs:
            if has_word(model, w1) and has_word(model, w2):
                print(f"{w1:20s} <-> {w2:20s}: {cosine(model, w1, w2):.4f}")
            else:
                print(f"{w1:20s} <-> {w2:20s}: OOV")

        trained_models.append((model, window, rel_results, analogy_results))

    print(f"\n{'='*45}")
    print("  COMPARISON: window=5 vs window=10")
    print(f"{'='*45}")

    model_w5,  _, rel_w5,  analogy_w5  = trained_models[0]
    model_w10, _, rel_w10, analogy_w10 = trained_models[1]

    w5_map  = {(w1, w2): pred for w1, w2, _, pred in rel_w5["covered_items"]}
    w10_map = {(w1, w2): pred for w1, w2, _, pred in rel_w10["covered_items"]}
    gold_map = {(w1, w2): gold for w1, w2, gold, _ in rel_w5["covered_items"]}
    gold_map.update({(w1, w2): gold for w1, w2, gold, _ in rel_w10["covered_items"]})

    print(f"\n{'Pair':<35} {'W5 pred':>10} {'W10 pred':>10} {'Gold':>8}")
    print("-" * 67)
    for pair in sorted(set(w5_map) | set(w10_map)):
        w1, w2 = pair
        label = f"{w1} - {w2}"
        print(f"{label:<35} {w5_map.get(pair, float('nan')):>10.4f} {w10_map.get(pair, float('nan')):>10.4f} {gold_map[pair]:>8.2f}")

    print(f"\n{'Metric':<30} {'window=5':>12} {'window=10':>12}")
    print("-" * 56)
    print(f"{'Relatedness coverage':<30} {rel_w5['coverage']:>9}/{rel_w5['total']} {rel_w10['coverage']:>9}/{rel_w10['total']}")
    print(f"{'Analogy coverage':<30} {analogy_w5['coverage']:>9}/{analogy_w5['total']} {analogy_w10['coverage']:>9}/{analogy_w10['total']}")
    print(f"{'Analogy top-5 accuracy':<30} {analogy_w5['accuracy_top5']:>12.4f} {analogy_w10['accuracy_top5']:>12.4f}")

    pca_words = [
        "radicalization", "extremism", "algorithm", "recommendation",
        "platform", "social", "media", "online", "users", "content",
        "echo", "chamber", "filter", "bubble", "hate", "speech",
        "conspiracy", "theory", "internet", "youtube"
    ]
    plot_pca([(model_w5, 5), (model_w10, 10)], pca_words)

    print("\nDone.")


main()
