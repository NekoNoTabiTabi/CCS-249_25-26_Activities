"""
Task 3 (10 points): Evaluate the Embeddings Using a Test Set
Evaluate the quality of learned word embeddings on multiple test sets.

This task includes:
- Creating custom evaluation test sets
- Computing semantic relatedness between word pairs
- Testing analogy completion capability
- Computing direct similarity scores
"""

import pickle
from typing import List, Tuple, Dict, Any
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import json


def has_word(model: Word2Vec, word: str) -> bool:
    """Check if a word exists in the model vocabulary."""
    return word in model.wv.key_to_index


def cosine_similarity_score(model: Word2Vec, w1: str, w2: str) -> float:
    """
    Compute cosine similarity between two word vectors.
    
    Args:
        model: Trained Word2Vec model
        w1: First word
        w2: Second word
        
    Returns:
        Cosine similarity score (0 to 1, where 1 = identical direction)
    """
    v1 = model.wv[w1].reshape(1, -1)
    v2 = model.wv[w2].reshape(1, -1)
    return float(cosine_similarity(v1, v2)[0][0])


def evaluate_relatedness(model: Word2Vec, 
                         test_pairs: List[Tuple[str, str, float]]) -> Dict[str, Any]:
    """
    TASK 3a: Evaluate embeddings using semantic relatedness test set.
    
    This test measures whether the model correctly captures semantic
    relationships between word pairs compared to human annotations.
    
    Args:
        model: Trained Word2Vec model
        test_pairs: List of (word1, word2, human_score) tuples
                   where human_score is expected relatedness (0-1)
        
    Returns:
        Dictionary with evaluation results
    """
    gold = []
    pred = []
    covered = []

    for w1, w2, score in test_pairs:
        if has_word(model, w1) and has_word(model, w2):
            sim = cosine_similarity_score(model, w1, w2)
            gold.append(score)
            pred.append(sim)
            covered.append((w1, w2, score, sim))

    return {
        "covered_items": covered,
        "coverage": len(covered),
        "total": len(test_pairs),
    }


def evaluate_analogies(model: Word2Vec, 
                       analogies: List[Tuple[str, str, str, str]]) -> Dict[str, Any]:
    """
    Evaluate word analogies (a:b :: c:d).
    
    Checks whether the model answer: vector(b) - vector(a) + vector(c) ≈ vector(d)
    
    Args:
        model: Trained Word2Vec model
        analogies: List of (a, b, c, d) tuples forming analogies
        
    Returns:
        Dictionary with analogy evaluation results
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


if __name__ == "__main__":
    print("=== TASK 3: Evaluate Embeddings Using Test Sets ===\n")
    
    # Load the trained model
    print("Loading trained model...")
    model = Word2Vec.load("skipgram_model.model")
    print(f"Loaded model with vocabulary size: {len(model.wv)}\n")
    
    # TASK 3b & 3c: Define custom evaluation test sets
    # You can change these based on what words appear in your corpus
    # and what relationships you want to test
    
    print("Creating relatedness test set...")
    # TASK 3b: Customize this test set with words from your dataset
    # Higher scores should indicate more semantic relatedness
    relatedness_test = [
        # JoJo's Bizarre Adventure related pairs
        ("jojo", "bizarre", 0.85),
        ("stand", "power", 0.80),
        ("stand", "ability", 0.80),
        ("jojo", "adventure", 0.75),
        ("bizarre", "adventure", 0.70),
        ("jojo", "series", 0.85),
        ("part", "episode", 0.60),
        ("character", "protagonist", 0.70),
        ("anime", "series", 0.85),
        ("manga", "story", 0.75),
        
        # Negative examples (unrelated words)
        ("jojo", "kitchen", 0.05),
        ("stand", "chair", 0.10),
        ("bizarre", "normal", 0.15),
        ("adventure", "boring", 0.10),
    ]
    
    rel_results = evaluate_relatedness(model, relatedness_test)
    
    print(f"Relatedness Test Coverage: {rel_results['coverage']}/{rel_results['total']}")
    print("\nRelatedness Test Results (TASK 3c):")
    for w1, w2, gold, pred in rel_results["covered_items"]:
        print(f"  {w1:15s} - {w2:15s} | expected={gold:.2f}, predicted={pred:.4f}")
    
    # Analogy test
    print("\n\nCreating analogy test set...")
    analogy_test = [
        ("jojo", "bizarre", "adventure", "series"),
        ("part", "episode", "arc", "story"),
        ("character", "protagonist", "villain", "antagonist"),
        ("stand", "ability", "power", "strength"),
    ]
    
    analogy_results = evaluate_analogies(model, analogy_test)
    
    print(f"Analogy Test Coverage: {analogy_results['coverage']}/{analogy_results['total']}")
    print(f"Top-5 Analogy Accuracy: {analogy_results['accuracy_top5']:.2%}")
    print("\nAnalogy Test Results (TASK 3c):")
    for item in analogy_results["details"]:
        print(f"  {item['analogy']:25s} -> predictions: {item['predictions']}")
    
    # Direct similarity checks for inspection
    print("\n\nDirect Similarity Checks:")
    check_pairs = [
        ("jojo", "bizarre"),
        ("stand", "power"),
        ("jojo", "kitchen"),
        ("adventure", "series"),
    ]
    
    for w1, w2 in check_pairs:
        if has_word(model, w1) and has_word(model, w2):
            sim = cosine_similarity_score(model, w1, w2)
            print(f"  {w1:15s} <-> {w2:15s}: {sim:.4f}")
        else:
            missing = []
            if not has_word(model, w1): missing.append(w1)
            if not has_word(model, w2): missing.append(w2)
            print(f"  {w1:15s} <-> {w2:15s}: OOV ({', '.join(missing)})")
    
    # Save results
    print("\n\nSaving evaluation results...")
    with open("evaluation_results.pkl", "wb") as f:
        pickle.dump({
            "relatedness": rel_results,
            "analogies": analogy_results
        }, f)
    print("Saved to: evaluation_results.pkl")
