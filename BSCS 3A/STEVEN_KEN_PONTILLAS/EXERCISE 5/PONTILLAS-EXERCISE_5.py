# Dorado, Louise Marielle V. & Pontillas, Steven Ken E.
# Exercise 5: Skip-gram with Negative Sampling (SGNS) Training and Analysis

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
import numpy as np
import matplotlib.pyplot as plt


# Paste the Wikipedia link you want to use here.
# The article should be reasonably long (at least a few thousand words) for good results.
WIKI_URL = "https://en.wikipedia.org/wiki/Bigfoot"
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
        "User-Agent": "Mozilla/5.0 (compatible; SGNS-Bigfoot-Training/1.0)"
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
    """
    REQUIREMENT #3: Train a Skip-gram with Negative Sampling model
    This function trains the Word2Vec model with Skip-gram architecture.
    
    Parameters:
        window_size: context window size (default=5)
    """
    model = Word2Vec(
        sentences=sentences,
        vector_size=100, # What happens if we change this? Try 50, 200, 300 and see how it affects results.
        window=window_size,  # Modified to accept parameter
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


def visualize_embeddings_pca(model: Word2Vec, words: List[str], title: str = "Word Embeddings PCA Visualization"):
    """
    REQUIREMENT #5 (Part 2): Visualize the vectors using PCA
    Generates a 2D PCA visualization of word embeddings.
    """
    # Extract vectors for selected words
    vectors = []
    labels = []
    
    for word in words:
        if has_word(model, word):
            vectors.append(model.wv[word])
            labels.append(word)
    
    if len(vectors) < 2:
        print("Not enough words in vocabulary for visualization.")
        return
    
    # Apply PCA to reduce to 2D
    vectors = np.array(vectors)
    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    reduced_vectors = pca.fit_transform(vectors)
    
    # Create visualization
    plt.figure(figsize=(14, 10))
    plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], alpha=0.6, s=100, c=range(len(labels)), cmap='tab20')
    
    for i, label in enumerate(labels):
        plt.annotate(label, (reduced_vectors[i, 0], reduced_vectors[i, 1]), 
                    fontsize=10, ha='center', xytext=(5, 5), textcoords='offset points')
    
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt


def main():
    """
    NLP Activity: Training Skip-gram with Negative Sampling
    """
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    ensure_nltk()




    # 1. Use a Wikipedia article as the dataset
    print("Downloading Wikipedia article...")
    raw_text = fetch_wikipedia_article(WIKI_URL)  # Lines 37-50: fetch_wikipedia_article()



    # 2. Preprocess the text from the selected corpus
    print("Preprocessing text...")
    sentences = preprocess_text(raw_text)  # Lines 54-82: preprocess_text()
    stats = corpus_stats(sentences)

    print("\n=== Corpus Stats ===")
    for k, v in stats.items():
        print(f"{k}: {v}")




    # 3. Train Skip-gram with Negative Sampling (window=5)
    print("\n" + "="*60)
    print("MODEL 1: Skip-gram with window_size=5")
    print("="*60)
    print("Training Skip-gram with Negative Sampling (window=5)...")
    model = train_sgns(sentences, window_size=5)  # Lines 104-125: train_sgns()

    print("\nVocabulary size learned:", len(model.wv))
    print("\nKey Word2Vec properties:")
    print(f"  - Vector size: {model.wv.vector_size}")
    print(f"  - Window size: 5")
    print(f"  - Architecture: Skip-gram (sg=1)")
    print(f"  - Negative sampling: 10")

    probe_words = [
        "bigfoot", "sasquatch", "cryptid", "sighting", "forest",
        "footprint", "report", "evidence", "legend", "hoax"
    ]
    print_top_neighbors(model, probe_words, topn=8)  # Lines 189-199: print_top_neighbors()

    # Domain-specific relatedness test set
    relatedness_test = [
        ("bigfoot", "sasquatch", 0.95),
        ("bigfoot", "footprint", 0.85),
        ("bigfoot", "sighting", 0.85),
        ("cryptid", "legend", 0.80),
        ("report", "evidence", 0.70),
        ("forest", "wilderness", 0.75),
        ("bigfoot", "hoax", 0.20),
        ("sighting", "banana", 0.05),
    ]

    rel_results = evaluate_relatedness(model, relatedness_test)  # Lines 135-154: evaluate_relatedness()

    print("\n=== Relatedness Test Set (window=5) ===")
    print(f"Coverage: {rel_results['coverage']}/{rel_results['total']}")
    for w1, w2, gold, pred in rel_results["covered_items"]:
        print(f"{w1:10s} - {w2:10s} | gold={gold:.2f} pred={pred:.4f}")

    analogy_test = [
        ("bigfoot", "sasquatch", "legend", "cryptid"),
        ("report", "evidence", "sighting", "footprint"),
        ("forest", "wilderness", "story", "legend"),
    ]

    analogy_results = evaluate_analogies(model, analogy_test)  # Lines 158-186: evaluate_analogies()

    print("\n=== Analogy Test Set (window=5) ===")
    print(f"Coverage: {analogy_results['coverage']}/{analogy_results['total']}")
    print(f"Top-5 accuracy: {analogy_results['accuracy_top5']:.4f}")
    for item in analogy_results["details"]:
        print(json.dumps(item, ensure_ascii=False))





    # 4. Evaluate the embeddings using a small test set
    # Direct similarity checks
    print("\n=== Direct Similarity Checks (window=5) ===")
    check_pairs = [
        ("bigfoot", "sasquatch"),
        ("bigfoot", "footprint"),
        ("report", "evidence"),
        ("sighting", "banana"),
    ]
    for w1, w2 in check_pairs:
        if has_word(model, w1) and has_word(model, w2):
            print(f"{w1:10s} <-> {w2:10s}: {cosine(model, w1, w2):.4f}")
        else:
            print(f"{w1:10s} <-> {w2:10s}: OOV")

    # Save model
    model.save("exercise_5_skipgram_sgns_window5.model")
    print("\nSaved model to: exercise_5_skipgram_sgns_window5.model")




    # 3. Retrain the model with window_size=10
    print("\n" + "="*60)
    print("MODEL 2: Skip-gram with window_size=10")
    print("="*60)
    print("Training Skip-gram with Negative Sampling (window=10)...")
    model_window10 = train_sgns(sentences, window_size=10)  # NEW: window=10

    print("\nVocabulary size learned:", len(model_window10.wv))
    print("\nKey Word2Vec properties:")
    print(f"  - Vector size: {model_window10.wv.vector_size}")
    print(f"  - Window size: 10 (CHANGED from 5)")
    print(f"  - Architecture: Skip-gram (sg=1)")
    print(f"  - Negative sampling: 10")

    print_top_neighbors(model_window10, probe_words, topn=8)

    # Evaluate with same test set
    rel_results_w10 = evaluate_relatedness(model_window10, relatedness_test)

    print("\n=== Relatedness Test Set (window=10) ===")
    print(f"Coverage: {rel_results_w10['coverage']}/{rel_results_w10['total']}")
    for w1, w2, gold, pred in rel_results_w10["covered_items"]:
        print(f"{w1:10s} - {w2:10s} | gold={gold:.2f} pred={pred:.4f}")

    analogy_results_w10 = evaluate_analogies(model_window10, analogy_test)

    print("\n=== Analogy Test Set (window=10) ===")
    print(f"Coverage: {analogy_results_w10['coverage']}/{analogy_results_w10['total']}")
    print(f"Top-5 accuracy: {analogy_results_w10['accuracy_top5']:.4f}")
    for item in analogy_results_w10["details"]:
        print(json.dumps(item, ensure_ascii=False))

    print("\n=== Direct Similarity Checks (window=10) ===")
    for w1, w2 in check_pairs:
        if has_word(model_window10, w1) and has_word(model_window10, w2):
            print(f"{w1:10s} <-> {w2:10s}: {cosine(model_window10, w1, w2):.4f}")
        else:
            print(f"{w1:10s} <-> {w2:10s}: OOV")

    model_window10.save("exercise_5_skipgram_sgns_window10.model")
    print("\nSaved model to: exercise_5_skipgram_sgns_window10.model")




    # 5. Report nearest neighbors, similarity scores, and test-set performance
    print("\n" + "="*60)
    print("COMPARISON ANALYSIS: Window Size Effect")
    print("="*60)
    print("\nSimilarity Score Comparison (select pairs):")
    print(f"{'Word Pair':<25} {'Window=5':<12} {'Window=10':<12} {'Difference':<12}")
    print("-" * 61)
    for w1, w2 in check_pairs:
        if has_word(model, w1) and has_word(model, w2) and \
           has_word(model_window10, w1) and has_word(model_window10, w2):
            sim5 = cosine(model, w1, w2)
            sim10 = cosine(model_window10, w1, w2)
            diff = sim10 - sim5
            print(f"{w1}-{w2:<20} {sim5:<12.4f} {sim10:<12.4f} {diff:+.4f}")

    print("\n=== Key Findings ===")
    print("Derivation from window size change (5 → 10):")
    print("- Larger window captures more distant contextual relationships")
    print("- May increase similarity between loosely related words")
    print("- Can improve rare word embeddings with more training context")
    print("- Might reduce discriminative power for closely related words")





    # 2. Visualize the vectors using PCA
    print("\n" + "="*60)
    print("VISUALIZATION: PCA of Word Embeddings")
    print("="*60)
    
    # Select at least 20 known words for visualization
    visualization_words = [
        "bigfoot", "sasquatch", "cryptid", "sighting", "forest",
        "footprint", "report", "evidence", "legend", "hoax",
        "wilderness", "creature", "story", "search", "witness",
        "track", "discovery", "mystery", "investigation", "myth",
        "expedition", "researcher", "hypothesis"
    ]

    print("\nGenerating PCA visualization for Model (window=5)...")
    plt_w5 = visualize_embeddings_pca(model, visualization_words, 
                                      "Word Embeddings - Skip-gram (window=5)")
    plt_w5.savefig("skipgram_pca_window5.png", dpi=150, bbox_inches='tight')
    print("Saved: skipgram_pca_window5.png")

    print("\nGenerating PCA visualization for Model (window=10)...")
    plt_w10 = visualize_embeddings_pca(model_window10, visualization_words, 
                                       "Word Embeddings - Skip-gram (window=10)")
    plt_w10.savefig("skipgram_pca_window10.png", dpi=150, bbox_inches='tight')
    print("Saved: skipgram_pca_window10.png")

    print("\nDone.")

main()