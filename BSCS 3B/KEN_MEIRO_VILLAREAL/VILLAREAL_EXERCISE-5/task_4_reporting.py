"""
Task 4 (10 points): Report Results
Generate comprehensive reports on:
- Nearest neighbors for selected probe words
- Similarity scores between word pairs
- Test-set performance (relatedness and analogies)
- Overall model quality assessment
"""

import pickle
import json
from typing import List
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr
import numpy as np


def has_word(model: Word2Vec, word: str) -> bool:
    """Check if a word exists in the model vocabulary."""
    return word in model.wv.key_to_index


def cosine_similarity_score(model: Word2Vec, w1: str, w2: str) -> float:
    """Compute cosine similarity between two word vectors."""
    v1 = model.wv[w1].reshape(1, -1)
    v2 = model.wv[w2].reshape(1, -1)
    return float(cosine_similarity(v1, v2)[0][0])


def print_nearest_neighbors(model: Word2Vec, words: List[str], topn: int = 8):
    """
    TASK 4a: Report nearest neighbors for probe words.
    
    For each word, finds the topn most similar words in the embedding space.
    
    Args:
        model: Trained Word2Vec model
        words: List of probe words to examine
        topn: Number of neighbors to report
    """
    print("\n" + "="*70)
    print("TASK 4a: NEAREST NEIGHBORS FOR PROBE WORDS")
    print("="*70)
    
    for word in words:
        if has_word(model, word):
            neighbors = model.wv.most_similar(word, topn=topn)
            print(f"\n[{word}]")
            for i, (neigh, score) in enumerate(neighbors, 1):
                print(f"  {i:2d}. {neigh:20s} (similarity: {score:.4f})")
        else:
            print(f"\n[{word}] - OUT OF VOCABULARY")


def print_similarity_scores(model: Word2Vec, pairs: List[tuple]):
    """
    TASK 4a: Report similarity scores for word pairs.
    
    Computes and displays cosine similarity for given word pairs.
    
    Args:
        model: Trained Word2Vec model
        pairs: List of (word1, word2) tuples
    """
    print("\n" + "="*70)
    print("TASK 4a: DIRECT SIMILARITY SCORES")
    print("="*70)
    
    print(f"\n{'Word 1':15s} {'Word 2':15s} {'Similarity':15s} {'Status'}")
    print("-" * 60)
    
    for w1, w2 in pairs:
        if has_word(model, w1) and has_word(model, w2):
            sim = cosine_similarity_score(model, w1, w2)
            status = "✓ Both in vocab"
            print(f"{w1:15s} {w2:15s} {sim:15.4f} {status}")
        else:
            missing = []
            if not has_word(model, w1): missing.append(w1)
            if not has_word(model, w2): missing.append(w2)
            status = f"OOV: {', '.join(missing)}"
            print(f"{w1:15s} {w2:15s} {'N/A':15s} {status}")


def print_test_set_performance(rel_results: dict, analogy_results: dict):
    """
    TASK 4a: Report test set performance metrics.
    
    Includes:
    - Coverage on relatedness task
    - Correlation between human scores and model predictions
    - Accuracy on analogy task
    
    Args:
        rel_results: Results from relatedness evaluation
        analogy_results: Results from analogy evaluation
    """
    print("\n" + "="*70)
    print("TASK 4a: TEST SET PERFORMANCE")
    print("="*70)
    
    # Relatedness performance
    print("\n[RELATEDNESS TEST SET]")
    print(f"  Coverage: {rel_results['coverage']}/{rel_results['total']} "
          f"({100*rel_results['coverage']/rel_results['total']:.1f}%)")
    
    if rel_results['covered_items']:
        gold_scores = [item[2] for item in rel_results['covered_items']]
        pred_scores = [item[3] for item in rel_results['covered_items']]
        
        if len(set(pred_scores)) > 1:  # Only compute if not all predictions are identical
            corr, pval = spearmanr(gold_scores, pred_scores)
            print(f"  Spearman Correlation: {corr:.4f} (p-value: {pval:.4f})")
        
        mae = np.mean([abs(g - p) for g, p in zip(gold_scores, pred_scores)])
        print(f"  Mean Absolute Error: {mae:.4f}")
        
        print(f"\n  {'Pair':30s} {'Expected':10s} {'Predicted':10s} {'Error'}")
        print(f"  {'-'*60}")
        for w1, w2, gold, pred in rel_results['covered_items']:
            pair_str = f"{w1}-{w2}"
            error = abs(gold - pred)
            print(f"  {pair_str:30s} {gold:10.2f} {pred:10.4f} {error:6.4f}")
    
    # Analogy performance
    print("\n[ANALOGY TEST SET]")
    print(f"  Coverage: {analogy_results['coverage']}/{analogy_results['total']} "
          f"({100*analogy_results['coverage']/analogy_results['total']:.1f}%)")
    print(f"  Top-5 Accuracy: {analogy_results['accuracy_top5']:.2%}")
    
    if analogy_results['details']:
        print(f"\n  {'Analogy':30s} {'Expected':15s} {'Predictions (Top-5)'}")
        print(f"  {'-'*70}")
        for item in analogy_results['details']:
            preds_str = ", ".join(item['predictions'][:3])
            match = "✓" if item['correct_in_top5'] else "✗"
            print(f"  {item['analogy']:30s} {item['expected']:15s} {match} {preds_str}")


def print_summary_report(model: Word2Vec, rel_results: dict, analogy_results: dict):
    """
    TASK 4a: Print comprehensive summary report.
    """
    print("\n" + "="*70)
    print("SUMMARY: SKIP-GRAM MODEL QUALITY ASSESSMENT")
    print("="*70)
    
    print(f"\n[Model Configuration]")
    print(f"  Vector Dimension:  {model.wv.vector_size}")
    print(f"  Vocabulary Size:   {len(model.wv)}")
    print(f"  Training Epochs:   {model.epochs}")
    
    print(f"\n[Dataset Statistics]")
    # Note: These are estimated from the model
    print(f"  Total unique words learned: {len(model.wv)}")
    
    print(f"\n[Evaluation Metrics]")
    rel_cov_pct = 100 * rel_results['coverage'] / rel_results['total']
    analogy_cov_pct = 100 * analogy_results['coverage'] / analogy_results['total']
    
    print(f"  Relatedness Coverage:    {rel_cov_pct:6.1f}%")
    print(f"  Analogy Coverage:        {analogy_cov_pct:6.1f}%")
    print(f"  Analogy Top-5 Accuracy:  {analogy_results['accuracy_top5']:6.1%}")
    
    print(f"\n[Observations]")
    if rel_results['coverage'] < 0.5 * rel_results['total']:
        print(f"  ⚠ Low coverage on relatedness test - many words out of vocabulary")
    else:
        print(f"  ✓ Good coverage on relatedness test")
    
    if analogy_results['accuracy_top5'] > 0.5:
        print(f"  ✓ Model captures analogy relationships well")
    else:
        print(f"  ⚠ Model struggles with analogy relationships")
    
    print(f"\n[Interpretation]")
    print(f"  The skip-gram model has learned semantic relationships between words")
    print(f"  based on their co-occurrence patterns in the Wikipedia article.")
    print(f"  Words appearing in similar contexts have similar vector representations.")


if __name__ == "__main__":
    print("=== TASK 4: GENERATE RESULTS REPORT ===\n")
    
    # Load model and results
    print("Loading trained model...")
    model = Word2Vec.load("skipgram_model.model")
    
    print("Loading evaluation results...")
    with open("evaluation_results.pkl", "rb") as f:
        results = pickle.load(f)
    
    rel_results = results["relatedness"]
    analogy_results = results["analogies"]
    
    print("Model and results loaded successfully.\n")
    
    # TASK 4a: Report nearest neighbors
    probe_words = [
        "jojo", "bizarre", "stand", "power", "adventure",
        "anime", "series", "character", "episode", "manga"
    ]
    print_nearest_neighbors(model, probe_words, topn=8)
    
    # TASK 4a: Report similarity scores
    similarity_pairs = [
        ("jojo", "bizarre"),
        ("stand", "power"),
        ("jojo", "adventure"),
        ("anime", "series"),
        ("character", "protagonist"),
        ("jojo", "kitchen"),
        ("stand", "chair"),
    ]
    print_similarity_scores(model, similarity_pairs)
    
    # TASK 4a: Report test-set performance
    print_test_set_performance(rel_results, analogy_results)
    
    # Summary report
    print_summary_report(model, rel_results, analogy_results)
    
    print("\n" + "="*70)
    print("END OF REPORT")
    print("="*70 + "\n")
    
    # Save report to file
    report_data = {
        "model_vocab_size": len(model.wv),
        "model_vector_size": model.wv.vector_size,
        "relatedness_results": rel_results,
        "analogy_results": analogy_results,
    }
    
    with open("final_report.json", "w") as f:
        json.dump(report_data, f, indent=2, default=str)
    
    print("Report saved to: final_report.json")
