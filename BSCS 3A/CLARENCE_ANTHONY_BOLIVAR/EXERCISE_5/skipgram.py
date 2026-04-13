"""
Train Skip-gram with Negative Sampling on a Wikipedia article,
then evaluate the embedding model with intrinsic tests and custom test sets.

Requirements:
    pip install requests beautifulsoup4 nltk gensim scikit-learn scipy matplotlib
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── Step 1: Dataset ──────────────────────────────────────────────────────────
# Wikipedia article used as the corpus.
# Ikigai is a reasonably long article with rich Japanese life-philosophy vocabulary.
WIKI_URL = "https://en.wikipedia.org/wiki/Ikigai"
RANDOM_SEED = 42


def ensure_nltk():
    resources = ["punkt", "punkt_tab"]
    for r in resources:
        try:
            nltk.data.find(f"tokenizers/{r}")
        except LookupError:
            nltk.download(r)


# ── Step 1 (codeline): fetch_wikipedia_article ────────────────────────────
def fetch_wikipedia_article(url: str) -> str:
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; SGNS-Gundam-Training/1.0)"
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


# ── Step 2 (codeline): preprocess_text ────────────────────────────────────
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


# ── Step 3 (codeline): train_sgns ────────────────────────────────────────
def train_sgns(sentences: List[List[str]], window: int = 5) -> Word2Vec:
    """
    Key Word2Vec parameters:
      vector_size : dimensionality of the embedding vectors (100)
      window      : context window — how many words left/right to consider (5 or 10)
      sg=1        : use Skip-gram (0 = CBOW)
      negative=10 : number of negative samples per positive pair
    """
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


def cosine_sim(model: Word2Vec, w1: str, w2: str) -> float:
    v1 = model.wv[w1].reshape(1, -1)
    v2 = model.wv[w2].reshape(1, -1)
    return float(cosine_similarity(v1, v2)[0][0])


# ── Step 4 (codeline): evaluate_relatedness / evaluate_analogies ──────────
def evaluate_relatedness(model: Word2Vec, test_pairs: List[Tuple[str, str, float]]):
    gold, pred, covered = [], [], []
    for w1, w2, score in test_pairs:
        if has_word(model, w1) and has_word(model, w2):
            sim = cosine_sim(model, w1, w2)
            gold.append(score)
            pred.append(sim)
            covered.append((w1, w2, score, sim))
    return {"covered_items": covered, "coverage": len(covered), "total": len(test_pairs)}


def evaluate_analogies(model: Word2Vec, analogies: List[Tuple[str, str, str, str]]):
    """
    Analogy format a:b :: c:?
    Predicts d via most_similar(positive=[b, c], negative=[a]).
    """
    covered, correct, details = 0, 0, []
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
    return {"coverage": covered, "total": len(analogies), "accuracy_top5": accuracy, "details": details}


# ── Step 5 (codeline): print_top_neighbors / similarity checks ────────────
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


def run_evaluation(model: Word2Vec, label: str):
    """Run the full evaluation suite and print results."""
    print(f"\n{'='*60}")
    print(f"  EVALUATION — {label}")
    print(f"{'='*60}")

    probe_words = [
        "ikigai", "purpose", "meaning", "life", "japanese",
        "happiness", "joy", "work", "passion", "reason"
    ]
    print_top_neighbors(model, probe_words, topn=8)

    # Domain-specific relatedness test set (Ikigai article)
    # Scores reflect expected semantic closeness based on the article's content.
    relatedness_test = [
        ("purpose",   "meaning",   0.95),
        ("happiness", "joy",       0.90),
        ("work",      "passion",   0.80),
        ("life",      "reason",    0.75),
        ("japanese",  "culture",   0.85),
        ("purpose",   "reason",    0.85),
        ("ikigai",    "happiness", 0.80),
        ("meaning",   "joy",       0.70),
        ("ikigai",    "computer",  0.05),
        ("happiness", "rocket",    0.02),
        ("work",      "ocean",     0.05),
        ("passion",   "purpose",   0.85),
    ]

    rel_results = evaluate_relatedness(model, relatedness_test)
    print(f"\n=== Relatedness Test Set ===")
    print(f"Coverage: {rel_results['coverage']}/{rel_results['total']}")
    for w1, w2, gold, pred in rel_results["covered_items"]:
        print(f"  {w1:10s} - {w2:10s} | gold={gold:.2f}  pred={pred:.4f}")

    analogy_test = [
        ("happiness", "joy",     "sadness",   "sorrow"),
        ("work",      "passion", "life",      "purpose"),
        ("japan",     "japanese","france",    "french"),
        ("live",      "life",    "exist",     "existence"),
    ]

    analogy_results = evaluate_analogies(model, analogy_test)
    print(f"\n=== Analogy Test Set ===")
    print(f"Coverage: {analogy_results['coverage']}/{analogy_results['total']}")
    print(f"Top-5 accuracy: {analogy_results['accuracy_top5']}")
    for item in analogy_results["details"]:
        print(json.dumps(item, ensure_ascii=False))

    # Direct similarity checks
    print(f"\n=== Direct Similarity Checks ===")
    check_pairs = [
        ("purpose",   "meaning"),
        ("ikigai",    "happiness"),
        ("work",      "passion"),
        ("ikigai",    "computer"),
        ("happiness", "joy"),
        ("life",      "reason"),
    ]
    for w1, w2 in check_pairs:
        if has_word(model, w1) and has_word(model, w2):
            print(f"  {w1:12s} <-> {w2:12s}: {cosine_sim(model, w1, w2):.4f}")
        else:
            print(f"  {w1:12s} <-> {w2:12s}: OOV")

    return rel_results, analogy_results


# ── PCA Visualization ────────────────────────────────────────────────────
def plot_pca(model: Word2Vec, filename: str = "pca_vectors.png"):
    """
    Project the top 30 most-frequent in-vocabulary Ikigai words down to 2D
    via PCA and save a scatter plot.
    """
    target_words = [
        "ikigai", "purpose", "meaning", "life", "reason",
        "happiness", "joy", "work", "passion", "pleasure",
        "japanese", "culture", "japan", "okinawa", "longevity",
        "concept", "existence", "satisfaction", "fulfillment", "worth",
        "health", "wellbeing", "motivation", "value", "sense",
        "living", "individual", "social", "mental", "emotional",
    ]

    # Keep only words the model actually knows
    known = [w for w in target_words if has_word(model, w)]
    if len(known) < 3:
        print("Not enough known words for PCA plot.")
        return

    vectors = np.array([model.wv[w] for w in known])

    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    coords = pca.fit_transform(vectors)

    fig, ax = plt.subplots(figsize=(12, 9))
    ax.scatter(coords[:, 0], coords[:, 1], s=60, color="steelblue", zorder=3)

    for i, word in enumerate(known):
        ax.annotate(
            word,
            xy=(coords[i, 0], coords[i, 1]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=10,
        )

    ax.set_title("PCA Projection of Word2Vec Embeddings (Ikigai article)", fontsize=13)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"\nPCA plot saved to: {filename}")
    print(f"Words plotted ({len(known)}): {', '.join(known)}")


def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    ensure_nltk()

    # ── Step 1: Download article ───────────────────────────────────────────
    print("Downloading Wikipedia article:", WIKI_URL)
    raw_text = fetch_wikipedia_article(WIKI_URL)

    # ── Step 2: Preprocess ────────────────────────────────────────────────
    print("Preprocessing text...")
    sentences = preprocess_text(raw_text)
    stats = corpus_stats(sentences)

    print("\n=== Corpus Stats ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # ── Step 3 (OLD): Train with window=5 ─────────────────────────────────
    print("\nTraining Skip-gram (window=5) ...")
    model_w5 = train_sgns(sentences, window=5)
    print("Vocabulary size learned:", len(model_w5.wv))

    # ── Steps 4 & 5: Evaluate OLD model ───────────────────────────────────
    run_evaluation(model_w5, "OLD — window=5")

    model_w5.save("model_window5.model")
    print("\nSaved OLD model to: model_window5.model")

    # ── Step 3 (NEW): Retrain with window=10 ──────────────────────────────
    print("\n\nRetraining Skip-gram (window=10) ...")
    model_w10 = train_sgns(sentences, window=10)
    print("Vocabulary size learned:", len(model_w10.wv))

    # ── Steps 4 & 5: Evaluate NEW model ───────────────────────────────────
    run_evaluation(model_w10, "NEW — window=10")

    model_w10.save("model_window10.model")
    print("\nSaved NEW model to: model_window10.model")

    # ── PCA Visualization (window=10 model) ───────────────────────────────
    print("\nGenerating PCA plot...")
    plot_pca(model_w10, filename="pca_vectors.png")

    print("\nDone.")


main()
