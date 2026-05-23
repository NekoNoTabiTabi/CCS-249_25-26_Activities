"""
Train Skip-gram with Negative Sampling on a Wikipedia article,
then evaluate the embedding model with intrinsic tests and custom test sets.

Flow (see main()):
  1. Download article HTML from WIKI_URL and extract main body text.
  2. Preprocess into tokenized sentences for gensim.
  3. Train Word2Vec with skip-gram + negative sampling (train_sgns).
     Trains twice: window=5 (baseline) vs window=10 (assignment); prints metric comparison.
  4. Inspect neighbors (probe_words), relatedness pairs, analogies, cosine checks (window=10 model).
  5. Save the window=10 model to exercise_5_skipgram_sgns.model next to this script.
  6. PCA 2D plot of >=20 word vectors → word2vec_pca_star_wars.png (paste into your report).

Run from this folder:
    python VALLOTA-UNIT5-EXERCISE.py

Requirements:
    pip install requests beautifulsoup4 nltk gensim scikit-learn scipy matplotlib

Optional:
    python -m nltk.downloader punkt punkt_tab stopwords
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


WIKI_URL = "https://en.wikipedia.org/wiki/Star_Wars"
RANDOM_SEED = 42

# Extra Star Wars–flavored tokens for PCA (must be in model vocab after training).
PCA_EXTRA_WORDS = [
    "yoda", "darth", "rebel", "rebels", "sith", "alien", "planet", "episode",
    "lucas", "space", "character", "novel", "comic", "animation", "video",
    "game", "prequel", "sequel", "trilogy", "lightsaber",
]
PCA_MIN_WORDS = 20
PCA_MAX_WORDS = 32
PCA_OUT_FILE = "word2vec_pca_star_wars.png"


def ensure_nltk():
    resources = ["punkt", "punkt_tab"]
    for r in resources:
        try:
            nltk.data.find(f"tokenizers/{r}")
        except LookupError:
            nltk.download(r)


def fetch_wikipedia_article(url: str) -> str:
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; SGNS-WikiTraining/1.0; educational)"
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


def train_sgns(sentences: List[List[str]], window: int = 10) -> Word2Vec:
    model = Word2Vec(
        sentences=sentences,
        vector_size=200, # What happens if we change this? Try 50, 200, 300 and see how it affects results.
        window=window,
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


def evaluation_metrics(
    model: Word2Vec,
    relatedness_test: List[Tuple[str, str, float]],
    analogy_test: List[Tuple[str, str, str, str]],
    check_pairs: List[Tuple[str, str]],
    probe_words: List[str],
    neighbor_topn: int = 8,
) -> Dict:
    """Scalar summaries for comparing runs (e.g. different window sizes)."""
    rel = evaluate_relatedness(model, relatedness_test)
    covered = rel["covered_items"]
    if covered:
        rel_mae = float(np.mean([abs(g - p) for _, _, g, p in covered]))
    else:
        rel_mae = float("nan")

    ana = evaluate_analogies(model, analogy_test)

    direct_scores = []
    for w1, w2 in check_pairs:
        if has_word(model, w1) and has_word(model, w2):
            direct_scores.append(cosine(model, w1, w2))
    mean_direct = float(np.mean(direct_scores)) if direct_scores else float("nan")

    nb_means = []
    for w in probe_words:
        if has_word(model, w):
            n = model.wv.most_similar(w, topn=neighbor_topn)
            nb_means.append(float(np.mean([s for _, s in n])))
    mean_neighbor_sim = float(np.mean(nb_means)) if nb_means else float("nan")

    return {
        "vocab_size": len(model.wv),
        "relatedness_coverage": rel["coverage"],
        "relatedness_total": rel["total"],
        "relatedness_mae_gold_vs_cosine": rel_mae,
        "analogy_coverage": ana["coverage"],
        "analogy_total": ana["total"],
        "analogy_accuracy_top5": ana["accuracy_top5"],
        "mean_cosine_check_pairs": mean_direct,
        "mean_top_neighbor_similarity": mean_neighbor_sim,
    }


def print_metrics_table(label: str, m: Dict) -> None:
    print(f"\n--- {label} ---")
    print(f"  vocab_size:                      {m['vocab_size']}")
    print(f"  relatedness coverage:            {m['relatedness_coverage']}/{m['relatedness_total']}")
    print(f"  relatedness MAE (gold vs cos):   {m['relatedness_mae_gold_vs_cosine']:.4f}")
    print(f"  analogy coverage:                {m['analogy_coverage']}/{m['analogy_total']}")
    acc = m["analogy_accuracy_top5"]
    print(f"  analogy top-5 accuracy:          {acc:.4f}" if acc == acc else "  analogy top-5 accuracy:          nan")
    print(f"  mean cosine (check_pairs):       {m['mean_cosine_check_pairs']:.4f}")
    print(f"  mean top-{8} neighbor similarity:   {m['mean_top_neighbor_similarity']:.4f}")


def collect_pca_words(
    model: Word2Vec,
    probe_words: List[str],
    relatedness_test: List[Tuple[str, str, float]],
    analogy_test: List[Tuple[str, str, str, str]],
    check_pairs: List[Tuple[str, str]],
) -> List[str]:
    """At least PCA_MIN_WORDS in-vocab types, ordered (curated first, then corpus frequency)."""
    ordered: List[str] = []
    seen = set()

    def try_add(w: str) -> None:
        if w in seen or not has_word(model, w):
            return
        seen.add(w)
        ordered.append(w)

    for w in probe_words:
        try_add(w)
    for t in relatedness_test:
        try_add(t[0])
        try_add(t[1])
    for quad in analogy_test:
        for w in quad:
            try_add(w)
    for a, b in check_pairs:
        try_add(a)
        try_add(b)
    for w in PCA_EXTRA_WORDS:
        try_add(w)

    for w in model.wv.index_to_key:
        if len(ordered) >= PCA_MAX_WORDS:
            break
        try_add(w)

    if len(ordered) < PCA_MIN_WORDS:
        raise ValueError(
            f"Need at least {PCA_MIN_WORDS} in-vocab words for PCA; got {len(ordered)}."
        )
    return ordered[:PCA_MAX_WORDS]


def save_pca_word_plot(model: Word2Vec, words: List[str], out_path: str) -> None:
    """Project word vectors to 2D with PCA and save a scatter plot with labels."""
    X = np.vstack([model.wv[w] for w in words])
    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    xy = pca.fit_transform(X)

    plt.figure(figsize=(11, 9))
    plt.scatter(
        xy[:, 0], xy[:, 1], s=55, alpha=0.75, c="steelblue",
        edgecolors="black", linewidths=0.35,
    )
    for i, w in enumerate(words):
        plt.annotate(w, (xy[i, 0], xy[i, 1]), fontsize=8, ha="left", va="bottom", alpha=0.92)

    ev = pca.explained_variance_ratio_
    plt.xlabel(f"PC1 ({100 * ev[0]:.1f}% variance)")
    plt.ylabel(f"PC2 ({100 * ev[1]:.1f}% variance)")
    plt.title("Word2Vec embeddings (PCA, 2D)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    print(f"Saved PCA figure to: {out_path}")


def print_old_new_comparison(m_old: Dict, m_new: Dict, w_old: int, w_new: int) -> None:
    print("\n" + "=" * 60)
    print("WINDOW COMPARISON (fill OLD/NEW in your report from this table)")
    print("=" * 60)
    keys = [
        ("relatedness_mae_gold_vs_cosine", "Relatedness MAE (gold vs predicted cosine)"),
        ("analogy_accuracy_top5", "Analogy top-5 accuracy"),
        ("mean_cosine_check_pairs", "Mean cosine on check_pairs"),
        ("mean_top_neighbor_similarity", "Mean similarity of top-8 neighbors (probe words)"),
    ]
    print(f"{'Metric':<48} {'OLD w='+str(w_old):>14} {'NEW w='+str(w_new):>14}")
    print("-" * 78)
    for key, title in keys:
        a, b = m_old[key], m_new[key]
        sa = f"{a:.4f}" if isinstance(a, float) and a == a else str(a)
        sb = f"{b:.4f}" if isinstance(b, float) and b == b else str(b)
        print(f"{title:<48} {sa:>14} {sb:>14}")

    print("\n--- What to derive from changing window (typical patterns; confirm on your numbers) ---")
    print(
        "A larger window uses more distant context words when training skip-gram. "
        "Embeddings often become smoother or more topical: neighbor lists and cosine scores "
        "can shift. MAE on your hand-labeled relatedness pairs may go up or down depending "
        "on whether your gold judgments match 'wider context' co-occurrence in this one article. "
        "Analogy accuracy is noisy on a tiny test set—treat direction of change as illustrative, not definitive."
    )
    d_mae = m_new["relatedness_mae_gold_vs_cosine"] - m_old["relatedness_mae_gold_vs_cosine"]
    if d_mae == d_mae:
        if d_mae < -0.001:
            print(f"\nOn this run: NEW reduced relatedness MAE by {abs(d_mae):.4f} (cosines closer to your gold scores).")
        elif d_mae > 0.001:
            print(f"\nOn this run: NEW increased relatedness MAE by {d_mae:.4f} (cosines farther from your gold scores).")
        else:
            print("\nOn this run: relatedness MAE barely changed between windows.")


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

    WINDOW_OLD = 5
    WINDOW_NEW = 10

    probe_words = [
        "jedi", "empire", "skywalker", "vader", "droid",
        "franchise", "disney", "luke", "film", "series",
    ]

    # Domain-specific relatedness test set
    # Higher score means should be more semantically related
    relatedness_test = [
        # Gold scores: subjective targets for en.wikipedia.org/wiki/Star_Wars (same as WIKI_URL)—
        # high for core franchise / character / genre ties; low for unrelated pairs.
        ("star", "wars", 0.96),
        ("luke", "skywalker", 0.94),
        ("han", "solo", 0.92),
        ("clone", "trooper", 0.90),
        ("science", "fiction", 0.88),
        ("jedi", "force", 0.86),
        ("vader", "empire", 0.84),
        ("franchise", "disney", 0.80),
        ("original", "trilogy", 0.82),
        ("film", "series", 0.74),
        ("film", "movie", 0.90),
        ("jedi", "physics", 0.04),
        ("luke", "physics", 0.03),
        ("star", "physics", 0.03),
        ("film", "physics", 0.10),
    ]

    # Small analogy-style test set (a:b :: c:d  →  most_similar positive=[b,c] negative=[a])
    # Tiny on purpose: one Wikipedia page is a small corpus; swap words if coverage is low.
    analogy_test = [
        ("film", "movie", "television", "series"),
        ("empire", "jedi", "vader", "luke"),
        ("prequel", "trilogy", "sequel", "film"),
        ("han", "solo", "leia", "organa"),
    ]

    check_pairs = [
        ("luke", "skywalker"),
        ("jedi", "empire"),
        ("star", "wars"),
        ("jedi", "tractor"),
    ]

    print(f"\nTraining OLD: Skip-gram + negative sampling (window={WINDOW_OLD})...")
    model_old = train_sgns(sentences, window=WINDOW_OLD)
    metrics_old = evaluation_metrics(
        model_old, relatedness_test, analogy_test, check_pairs, probe_words
    )
    print_metrics_table(f"OLD model (window={WINDOW_OLD})", metrics_old)

    print(f"\nTraining NEW: Skip-gram + negative sampling (window={WINDOW_NEW})...")
    model_new = train_sgns(sentences, window=WINDOW_NEW)
    metrics_new = evaluation_metrics(
        model_new, relatedness_test, analogy_test, check_pairs, probe_words
    )
    print_metrics_table(f"NEW model (window={WINDOW_NEW})", metrics_new)

    print_old_new_comparison(metrics_old, metrics_new, WINDOW_OLD, WINDOW_NEW)

    model = model_new
    print("\n=== Full evaluation printout (NEW model, window=%d) ===" % WINDOW_NEW)
    print("\nVocabulary size learned:", len(model.wv))

    print_top_neighbors(model, probe_words, topn=8)

    rel_results = evaluate_relatedness(model, relatedness_test)
    print("\n=== Relatedness Test Set ===")
    print(f"Coverage: {rel_results['coverage']}/{rel_results['total']}")
    for w1, w2, gold, pred in rel_results["covered_items"]:
        print(f"{w1:10s} - {w2:10s} | gold={gold:.2f} pred={pred:.4f}")

    analogy_results = evaluate_analogies(model, analogy_test)
    print("\n=== Analogy Test Set ===")
    print(f"Coverage: {analogy_results['coverage']}/{analogy_results['total']}")
    print(f"Top-5 accuracy: {analogy_results['accuracy_top5']}")
    for item in analogy_results["details"]:
        print(json.dumps(item, ensure_ascii=False))

    print("\n=== Direct Similarity Checks ===")
    for w1, w2 in check_pairs:
        if has_word(model, w1) and has_word(model, w2):
            print(f"{w1:10s} <-> {w2:10s}: {cosine(model, w1, w2):.4f}")
        else:
            print(f"{w1:10s} <-> {w2:10s}: OOV")

    model.save("exercise_5_skipgram_sgns.model")
    print("\nSaved NEW model (window=%d) to: exercise_5_skipgram_sgns.model" % WINDOW_NEW)

    pca_words = collect_pca_words(
        model, probe_words, relatedness_test, analogy_test, check_pairs
    )
    print(f"\n=== PCA ({len(pca_words)} words) ===")
    print("Words in plot:", ", ".join(pca_words))
    save_pca_word_plot(model, pca_words, PCA_OUT_FILE)

    print("\nDone.")

if __name__ == "__main__":
    main()