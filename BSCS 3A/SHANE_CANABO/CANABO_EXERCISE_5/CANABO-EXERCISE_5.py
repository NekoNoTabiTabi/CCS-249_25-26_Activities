"""
Train and compare two Skip-gram with Negative Sampling models on
the West Visayas State University Wikipedia article.
"""

import re, math, json, random
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


WIKI_URL = "https://en.wikipedia.org/wiki/West_Visayas_State_University"
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
        "User-Agent": "Mozilla/5.0 (compatible; SGNS-WVSU-Training/1.0)"
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


def train_sgns(sentences: List[List[str]], vector_size: int, window: int, epochs: int) -> Word2Vec:
    return Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=1,
        workers=4,
        sg=1,
        negative=10,
        epochs=epochs,
        sample=1e-3,
        alpha=0.025,
        min_alpha=0.0007,
        seed=RANDOM_SEED,
    )


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
                    "correct_in_top5": hit,
                })
            except KeyError:
                pass

    accuracy = correct / covered if covered else float("nan")
    return {
        "coverage": covered,
        "total": len(analogies),
        "accuracy_top5": accuracy,
        "details": details,
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


def direct_similarity_checks(model: Word2Vec, pairs: List[Tuple[str, str]]) -> Dict[Tuple[str, str], float]:
    scores = {}
    print("\n=== Direct Similarity Checks ===")
    for w1, w2 in pairs:
        if has_word(model, w1) and has_word(model, w2):
            sim = cosine(model, w1, w2)
            scores[(w1, w2)] = sim
            print(f"{w1:12s} <-> {w2:12s}: {sim:.4f}")
        else:
            scores[(w1, w2)] = float("nan")
            print(f"{w1:12s} <-> {w2:12s}: OOV")
    return scores


def plot_pca_embeddings(model: Word2Vec, words: List[str], output_path: str = "pca_wvsu.png"):
    known_words = [w for w in words if has_word(model, w)]
    if len(known_words) < 2:
        print("\nNot enough known words for PCA plot.")
        return

    vectors = np.array([model.wv[w] for w in known_words])
    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    reduced = pca.fit_transform(vectors)

    plt.figure(figsize=(14, 10))
    plt.scatter(reduced[:, 0], reduced[:, 1], s=45, alpha=0.8)

    for i, word in enumerate(known_words):
        plt.annotate(word, (reduced[i, 0], reduced[i, 1]), fontsize=9)

    plt.title("WVSU Word Embeddings PCA (window=10)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"\nSaved PCA plot to: {output_path}")


def run_model_evaluations(model: Word2Vec, label: str, probe_words, relatedness_test, analogy_test, check_pairs):
    print(f"\n{'=' * 20} {label} {'=' * 20}")
    print(f"Vocabulary size learned: {len(model.wv)}")

    print_top_neighbors(model, probe_words, topn=8)

    rel_results = evaluate_relatedness(model, relatedness_test)
    print("\n=== Relatedness Test Set ===")
    print(f"Coverage: {rel_results['coverage']}/{rel_results['total']}")
    for w1, w2, gold, pred in rel_results["covered_items"]:
        print(f"{w1:12s} - {w2:12s} | gold={gold:.2f} pred={pred:.4f}")

    analogy_results = evaluate_analogies(model, analogy_test)
    print("\n=== Analogy Test Set ===")
    print(f"Coverage: {analogy_results['coverage']}/{analogy_results['total']}")
    print(f"Top-5 accuracy: {analogy_results['accuracy_top5']}")
    for item in analogy_results["details"]:
        print(json.dumps(item, ensure_ascii=False))

    sim_scores = direct_similarity_checks(model, check_pairs)

    return {
        "relatedness": rel_results,
        "similarity_scores": sim_scores,
    }


def print_comparison_summary(old_results, new_results, check_pairs):
    print("\n==================== OLD vs NEW Comparison ====================")
    old_rel = old_results["relatedness"]
    new_rel = new_results["relatedness"]
    print(
        f"Relatedness coverage | OLD: {old_rel['coverage']}/{old_rel['total']} "
        f"| NEW: {new_rel['coverage']}/{new_rel['total']}"
    )

    print("\nSimilarity scores (OLD vs NEW):")
    for pair in check_pairs:
        old_score = old_results["similarity_scores"][pair]
        new_score = new_results["similarity_scores"][pair]
        old_str = f"{old_score:.4f}" if not math.isnan(old_score) else "OOV"
        new_str = f"{new_score:.4f}" if not math.isnan(new_score) else "OOV"
        print(f"{pair[0]:12s} <-> {pair[1]:12s} | OLD: {old_str:>7s} | NEW: {new_str:>7s}")


def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    ensure_nltk()

    probe_words = [
        "wvsu", "university", "college", "students", "campus", "iloilo",
        "education", "faculty", "research", "science", "technology",
        "arts", "graduate", "undergraduate", "degree",
    ]

    relatedness_test = [
        ("university", "college", 0.90),
        ("students", "faculty", 0.70),
        ("campus", "university", 0.85),
        ("research", "science", 0.80),
        ("iloilo", "campus", 0.75),
        ("wvsu", "university", 0.95),
        ("education", "degree", 0.80),
        ("arts", "science", 0.60),
        ("wvsu", "kitchen", 0.05),
        ("university", "tractor", 0.02),
    ]

    analogy_test = [
        ("college", "university", "faculty", "students"),
        ("graduate", "degree", "undergraduate", "diploma"),
        ("science", "research", "arts", "culture"),
    ]

    check_pairs = [
        ("wvsu", "university"),
        ("students", "faculty"),
        ("research", "science"),
        ("wvsu", "kitchen"),
    ]

    pca_words = [
        "wvsu", "west", "visayas", "state", "university", "iloilo", "philippines",
        "college", "students", "faculty", "campus", "education", "research", "science",
        "technology", "arts", "graduate", "undergraduate", "degree", "program", "courses",
        "academic", "administration", "library", "extension",
    ]

    print("Downloading Wikipedia article...")
    raw_text = fetch_wikipedia_article(WIKI_URL)

    print("Preprocessing text...")
    sentences = preprocess_text(raw_text)
    stats = corpus_stats(sentences)
    token_counter = Counter([w for s in sentences for w in s])

    print("\n=== Corpus Stats ===")
    for k, v in stats.items():
        print(f"{k}: {v}")
    print("Top 10 frequent tokens:", token_counter.most_common(10))

    print("\nTraining Model 1 (OLD): vector_size=100, window=5, epochs=200")
    model_old = train_sgns(sentences, vector_size=100, window=5, epochs=200)

    print("\nTraining Model 2 (NEW): vector_size=100, window=10, epochs=200")
    model_new = train_sgns(sentences, vector_size=100, window=10, epochs=200)

    old_results = run_model_evaluations(
        model_old,
        "MODEL 1 (OLD | window=5)",
        probe_words,
        relatedness_test,
        analogy_test,
        check_pairs,
    )

    new_results = run_model_evaluations(
        model_new,
        "MODEL 2 (NEW | window=10)",
        probe_words,
        relatedness_test,
        analogy_test,
        check_pairs,
    )

    plot_pca_embeddings(model_new, pca_words, output_path="pca_wvsu.png")
    model_new.save("wvsu_skipgram_w10.model")
    print("Saved model to: wvsu_skipgram_w10.model")

    print_comparison_summary(old_results, new_results, check_pairs)
    print("\nDone.")


if __name__ == "__main__":
    main()