"""
Train Skip-gram with Negative Sampling on a Wikipedia article,
then evaluate the embedding model with intrinsic tests and custom test sets.

Requirements:
    pip install requests beautifulsoup4 nltk gensim scikit-learn scipy

Optional:
    python -m nltk.downloader punkt stopwords
"""

import json
import random
import re
from collections import Counter
from typing import Dict, List, Tuple

import nltk
import numpy as np
import requests
from bs4 import BeautifulSoup
from gensim.models import Word2Vec
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# ============================================================================
# Configuration Constants
# ============================================================================

# Wikipedia article URL for training corpus
# The article should be reasonably long (at least a few thousand words) for good results.
WIKI_URL = "https://en.wikipedia.org/wiki/Minecraft"

# Random seed for reproducibility
RANDOM_SEED = 42

# Text preprocessing constants
MIN_TOKEN_LENGTH = 2
MIN_SENTENCE_LENGTH = 3
WIKIPEDIA_TIMEOUT = 60
REQUEST_USER_AGENT = "Mozilla/5.0 (compatible; SGNS-Gundam-Training/1.0)"

# Model training constants
VECTOR_SIZE = 100
WINDOW_SIZE = 5
WINDOW_SIZE_COMPARISON = 10
MIN_COUNT = 1
NEGATIVE_SAMPLES = 10
NUM_EPOCHS = 200
INITIAL_ALPHA = 0.025
MIN_ALPHA = 0.0007
SAMPLE_THRESHOLD = 1e-3
NUM_WORKERS = 4
SKIP_GRAM = 1

# Evaluation constants
ANALOGY_TOPN = 5
NEIGHBOR_TOPN = 8

# Visualization constants
VISUALIZATION_WORDS = [
    "minecraft", "game", "player", "world", "block", "mob", "creeper",
    "nether", "end", "diamond", "gold", "crafting", "survival", "creative",
    "multiplayer", "server", "redstone", "biome", "armor", "weapon",
    "building", "mine", "skull", "history", "update", "version", "platform",
]


# ============================================================================
# NLTK Resource Management
# ============================================================================

def ensure_nltk() -> None:
    """
    Ensure required NLTK resources are downloaded.
    
    Downloads the 'punkt' and 'punkt_tab' tokenizers if not already available.
    """
    resources = ["punkt", "punkt_tab"]
    for resource in resources:
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            nltk.download(resource)


# ============================================================================
# Data Loading and Preprocessing
# ============================================================================

def prompt_for_wikipedia_file() -> str:
    """
    Prompt the user to select a local file containing Wikipedia content.

    Returns:
        The selected file path, or an empty string if the user cancels.
    """
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    filename = askopenfilename(
        title="Select Wikipedia content file",
        filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")],
    )
    root.destroy()
    return filename or ""


def fetch_wikipedia_article(url: str) -> str:
    """
    Fetch and extract text content from a Wikipedia article.
    
    Args:
        url: The Wikipedia URL to fetch
        
    Returns:
        Cleaned text content from the article
        
    Raises:
        ValueError: If the article content cannot be found
    """
    headers = {"User-Agent": REQUEST_USER_AGENT}
    response = requests.get(url, headers=headers, timeout=WIKIPEDIA_TIMEOUT)
    response.raise_for_status()

    # Extract main content text from the Wikipedia page
    soup = BeautifulSoup(response.text, "html.parser")
    content_div = soup.find("div", {"id": "mw-content-text"})
    
    if content_div is None:
        raise ValueError("Could not find Wikipedia article content.")

    # Collect text blocks from paragraphs and list items
    paragraphs = content_div.find_all(["p", "li"])
    text_blocks = []

    for paragraph in paragraphs:
        text = paragraph.get_text(" ", strip=True)
        if text:
            text_blocks.append(text)

    # Join all text blocks
    full_text = "\n".join(text_blocks)

    # Remove citation references and normalize whitespace
    full_text = re.sub(r"\[[0-9]+\]", " ", full_text)
    full_text = re.sub(r"\s+", " ", full_text).strip()
    
    return full_text


def load_wikipedia_content(url: str) -> str:
    """
    Load Wikipedia text from a local file or download it if no file is selected.
    """
    selected_file = prompt_for_wikipedia_file()
    if selected_file:
        with open(selected_file, "r", encoding="utf-8") as file:
            return file.read()
    return fetch_wikipedia_article(url)


def preprocess_text(text: str) -> List[List[str]]:
    """
    Preprocess raw text into tokenized sentences.
    
    Performs the following operations:
    - Sentence tokenization
    - Lowercase conversion
    - Removal of special characters
    - Word tokenization
    - Filtering of short/digit tokens
    
    Args:
        text: Raw text to preprocess
        
    Returns:
        List of tokenized sentences, where each sentence is a list of word tokens
    """
    sentences = sent_tokenize(text)
    processed_sentences = []

    for sentence in sentences:
        # Normalize and clean the sentence
        sentence = sentence.lower()
        sentence = re.sub(r"[^a-z0-9\-\s]", " ", sentence)
        sentence = re.sub(r"\s+", " ", sentence).strip()
        
        if not sentence:
            continue

        # Tokenize words
        tokens = word_tokenize(sentence)

        # Filter and clean tokens
        cleaned_tokens = []
        for token in tokens:
            token = token.strip("-")
            if not token:
                continue
            if token.isdigit():
                continue
            if len(token) < MIN_TOKEN_LENGTH:
                continue
            cleaned_tokens.append(token)

        # Keep sentences with minimum length
        if len(cleaned_tokens) >= MIN_SENTENCE_LENGTH:
            processed_sentences.append(cleaned_tokens)

    return processed_sentences


def corpus_stats(sentences: List[List[str]]) -> Dict[str, int]:
    """
    Calculate statistics about the corpus.
    
    Args:
        sentences: List of tokenized sentences
        
    Returns:
        Dictionary containing corpus statistics:
        - num_sentences: Total number of sentences
        - num_tokens: Total number of tokens
        - vocab_size: Number of unique vocabulary items
    """
    flat_tokens = [word for sentence in sentences for word in sentence]
    vocabulary = set(flat_tokens)
    
    return {
        "num_sentences": len(sentences),
        "num_tokens": len(flat_tokens),
        "vocab_size": len(vocabulary),
    }


# ============================================================================
# Model Training
# ============================================================================

def train_sgns(sentences: List[List[str]]) -> Word2Vec:
    """
    Train a Skip-gram with Negative Sampling (SGNS) word embedding model.
    
    Args:
        sentences: List of tokenized sentences for training
        
    Returns:
        Trained Word2Vec model
        
    Note:
        To experiment with vector sizes, try changing VECTOR_SIZE to 50, 200, or 300
        to see how it affects model performance and evaluation results.
    """
    model = Word2Vec(
        sentences=sentences,
        vector_size=VECTOR_SIZE,
        window=WINDOW_SIZE,
        min_count=MIN_COUNT,
        workers=NUM_WORKERS,
        sg=SKIP_GRAM,  # 0 = CBOW, 1 = Skip-gram
        negative=NEGATIVE_SAMPLES,
        epochs=NUM_EPOCHS,
        sample=SAMPLE_THRESHOLD,
        alpha=INITIAL_ALPHA,
        min_alpha=MIN_ALPHA,
        seed=RANDOM_SEED,
    )
    return model


def train_sgns_with_window_size(sentences: List[List[str]], window_size: int) -> Word2Vec:
    """
    Train the same SGNS model using a custom window size.

    Args:
        sentences: List of tokenized sentences for training
        window_size: Window size for context words

    Returns:
        Trained Word2Vec model with the requested window size
    """
    model = Word2Vec(
        sentences=sentences,
        vector_size=VECTOR_SIZE,
        window=window_size,
        min_count=MIN_COUNT,
        workers=NUM_WORKERS,
        sg=SKIP_GRAM,
        negative=NEGATIVE_SAMPLES,
        epochs=NUM_EPOCHS,
        sample=SAMPLE_THRESHOLD,
        alpha=INITIAL_ALPHA,
        min_alpha=MIN_ALPHA,
        seed=RANDOM_SEED,
    )
    return model


# ============================================================================
# Utility Functions
# ============================================================================

def has_word(model: Word2Vec, word: str) -> bool:
    """
    Check if a word exists in the model's vocabulary.
    
    Args:
        model: The Word2Vec model
        word: The word to check
        
    Returns:
        True if the word is in the vocabulary, False otherwise
    """
    return word in model.wv.key_to_index


def cosine_similarity_score(model: Word2Vec, word1: str, word2: str) -> float:
    """
    Calculate cosine similarity between two word vectors.
    
    Args:
        model: The Word2Vec model
        word1: First word
        word2: Second word
        
    Returns:
        Cosine similarity score between -1 and 1
    """
    vector1 = model.wv[word1].reshape(1, -1)
    vector2 = model.wv[word2].reshape(1, -1)
    return float(cosine_similarity(vector1, vector2)[0][0])


# ============================================================================
# Model Evaluation Functions
# ============================================================================

def evaluate_relatedness(
    model: Word2Vec, test_pairs: List[Tuple[str, str, float]]
) -> Dict:
    """
    Evaluate word embeddings on a word similarity task.
    
    Compares model's similarity scores with gold-standard relatedness scores.
    
    Args:
        model: The Word2Vec model
        test_pairs: List of (word1, word2, gold_score) tuples
        
    Returns:
        Dictionary containing:
        - covered_items: List of pairs that appear in vocabulary
        - coverage: Number of pairs covered
        - total: Total number of test pairs
    """
    gold_scores = []
    predicted_scores = []
    covered_items = []

    for word1, word2, gold_score in test_pairs:
        if has_word(model, word1) and has_word(model, word2):
            similarity = cosine_similarity_score(model, word1, word2)
            gold_scores.append(gold_score)
            predicted_scores.append(similarity)
            covered_items.append((word1, word2, gold_score, similarity))

    return {
        "covered_items": covered_items,
        "coverage": len(covered_items),
        "total": len(test_pairs),
    }


def evaluate_analogies(model: Word2Vec, analogies: List[Tuple[str, str, str, str]]) -> Dict:
    """
    Evaluate word embeddings on an analogy task.
    
    Analogy format: a:b :: c:d
    Checks whether most_similar(positive=[b,c], negative=[a]) returns d in the top-5.
    
    Args:
        model: The Word2Vec model
        analogies: List of (a, b, c, d) tuples representing analogies
        
    Returns:
        Dictionary containing:
        - coverage: Number of analogies fully covered in vocabulary
        - total: Total number of analogies
        - accuracy_top5: Accuracy of top-5 predictions
        - details: List of detailed results for each analogy
    """
    covered = 0
    correct = 0
    details = []

    for word_a, word_b, word_c, word_d in analogies:
        # Check if all words are in vocabulary
        if all(has_word(model, w) for w in [word_a, word_b, word_c, word_d]):
            covered += 1
            try:
                # Get predictions using most_similar
                predictions = model.wv.most_similar(
                    positive=[word_b, word_c], negative=[word_a], topn=ANALOGY_TOPN
                )
                predicted_words = [word for word, _ in predictions]
                is_correct = word_d in predicted_words
                correct += int(is_correct)
                
                details.append({
                    "analogy": f"{word_a}:{word_b}::{word_c}:?",
                    "expected": word_d,
                    "predictions": predicted_words,
                    "correct_in_top5": is_correct
                })
            except KeyError:
                pass

    # Calculate accuracy
    accuracy = correct / covered if covered else float("nan")
    
    return {
        "coverage": covered,
        "total": len(analogies),
        "accuracy_top5": accuracy,
        "details": details
    }


def print_top_neighbors(model: Word2Vec, words: List[str], topn: int = NEIGHBOR_TOPN) -> None:
    """
    Print the nearest neighbors for a list of words.
    
    Args:
        model: The Word2Vec model
        words: List of words to find neighbors for
        topn: Number of nearest neighbors to display (default: NEIGHBOR_TOPN)
    """
    print("\n=== Nearest Neighbors ===")
    for word in words:
        if has_word(model, word):
            neighbors = model.wv.most_similar(word, topn=topn)
            print(f"\n{word}:")
            for neighbor, score in neighbors:
                print(f"  {neighbor:20s} {score:.4f}")
        else:
            print(f"\n{word}: [OOV]")


def visualize_word_vectors(model: Word2Vec, words: List[str], title: str) -> None:
    """
    Visualize word vectors using PCA in two dimensions.
    
    Args:
        model: The Word2Vec model
        words: Words to visualize
        title: Plot title
    """
    word_list = [word for word in words if has_word(model, word)]
    if len(word_list) < 2:
        print("Not enough words are available in the vocabulary to visualize.")
        return

    vectors = np.array([model.wv[word] for word in word_list])
    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    embedding = pca.fit_transform(vectors)

    plt.figure(figsize=(12, 9))
    plt.scatter(embedding[:, 0], embedding[:, 1], edgecolors="k", c="tab:blue")

    for i, word in enumerate(word_list):
        plt.annotate(word, (embedding[i, 0] + 0.02, embedding[i, 1] + 0.02))

    plt.title(title)
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()


# ============================================================================
# Main Execution
# ============================================================================

def main() -> None:
    """
    Main execution function.
    
    Orchestrates the complete pipeline:
    1. Setup: Initialize random seeds and download NLTK resources
    2. Data Loading: Fetch Wikipedia article
    3. Preprocessing: Tokenize and clean text
    4. Training: Train Skip-gram with Negative Sampling model
    5. Analysis: Print nearest neighbors
    6. Evaluation: Run relatedness and analogy tests
    7. Export: Save trained model
    """
    # ========================================================================
    # Setup
    # ========================================================================
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    ensure_nltk()

    # ========================================================================
    # Data Loading and Preprocessing
    # ========================================================================
    print("Downloading Wikipedia article...")
    raw_text = load_wikipedia_content(WIKI_URL)

    print("Preprocessing text...")
    sentences = preprocess_text(raw_text)
    stats = corpus_stats(sentences)

    print("\n=== Corpus Stats ===")
    for key, value in stats.items():
        print(f"{key}: {value}")

    # ========================================================================
    # Model Training
    # ========================================================================
    print("\nTraining Skip-gram with Negative Sampling...")
    model = train_sgns(sentences)

    print(f"\nVocabulary size learned: {len(model.wv)}")

    # ========================================================================
    # Nearest Neighbors Analysis
    # ========================================================================
    probe_words = [
        "minecraft", "block", "survival", "creative", "mob",
        "crafting", "nether", "mojang", "microsoft", "world"
    ]
    print_top_neighbors(model, probe_words, topn=NEIGHBOR_TOPN)

    # ========================================================================
    # Relatedness Evaluation
    # ========================================================================
    
    # Domain-specific relatedness test set for Minecraft
    # Higher scores indicate words should be more semantically related in the article
    relatedness_test = [
        ("survival", "creative", 0.95),
        ("block", "voxel", 0.90),
        ("mojang", "microsoft", 0.85),
        ("nether", "dimension", 0.90),
        ("crafting", "recipe", 0.85),
        ("multiplayer", "server", 0.90),
        ("game", "player", 0.80),
        ("java", "bedrock", 0.85),
        ("creeper", "revenue", 0.05),
        ("block", "music", 0.10),
        ("survival", "publisher", 0.10),
        ("mob", "sales", 0.05),
    ]

    relatedness_results = evaluate_relatedness(model, relatedness_test)

    print("\n=== Relatedness Test Set ===")
    print(f"Coverage: {relatedness_results['coverage']}/{relatedness_results['total']}")
    for word1, word2, gold_score, pred_score in relatedness_results["covered_items"]:
        print(f"{word1:10s} - {word2:10s} | gold={gold_score:.2f} pred={pred_score:.4f}")

    # ========================================================================
    # Analogy Evaluation
    # ========================================================================
    
    # Small analogy-style test set.
    # These are intentionally small and corpus-dependent because a single article
    # is limited training data. Results may vary based on text preprocessing.
    analogy_test = [
        ("survival", "mode", "creative", "mode"),
        ("java", "pc", "bedrock", "console"),
        ("mojang", "developer", "microsoft", "publisher"),
        ("overworld", "dimension", "nether", "dimension"),  # May be OOV
    ]

    analogy_results = evaluate_analogies(model, analogy_test)

    print("\n=== Analogy Test Set ===")
    print(f"Coverage: {analogy_results['coverage']}/{analogy_results['total']}")
    print(f"Top-5 accuracy: {analogy_results['accuracy_top5']:.4f}")
    for item in analogy_results["details"]:
        print(json.dumps(item, ensure_ascii=False))

    # ========================================================================
    # Direct Similarity Checks
    # ========================================================================
    
    print("\n=== Direct Similarity Checks ===")
    # Modify these pairs based on expected relatedness in the article
    check_pairs = [
        ("survival", "creative"),
        ("mojang", "microsoft"),
        ("block", "voxel"),
        ("creeper", "sales"),  # Expected to show very low similarity
    ]
    
    for word1, word2 in check_pairs:
        if has_word(model, word1) and has_word(model, word2):
            similarity = cosine_similarity_score(model, word1, word2)
            print(f"{word1:10s} <-> {word2:10s}: {similarity:.4f}")
        else:
            print(f"{word1:10s} <-> {word2:10s}: OOV")


    # ========================================================================
    # Visualization
    # ========================================================================
    
    print("\n=== PCA Visualization ===")
    visualize_word_vectors(model, VISUALIZATION_WORDS, "Word2Vec PCA Visualization (Window 5)")

    # ========================================================================
    # Window Size Comparison
    # ========================================================================
    
    print(f"\nRetraining model with window size {WINDOW_SIZE_COMPARISON} for comparison...")
    model_window_10 = train_sgns_with_window_size(sentences, window_size=WINDOW_SIZE_COMPARISON)
    relatedness_results_window_10 = evaluate_relatedness(model_window_10, relatedness_test)
    analogy_results_window_10 = evaluate_analogies(model_window_10, analogy_test)

    print(f"\n=== Nearest Neighbors (Window {WINDOW_SIZE_COMPARISON}) ===")
    print_top_neighbors(model_window_10, probe_words, topn=NEIGHBOR_TOPN)

    print("\n=== Relatedness Test Set (Window 10) ===")
    print(f"Coverage: {relatedness_results_window_10['coverage']}/{relatedness_results_window_10['total']}")
    for word1, word2, gold_score, pred_score in relatedness_results_window_10["covered_items"]:
        print(f"{word1:10s} - {word2:10s} | gold={gold_score:.2f} pred={pred_score:.4f}")

    print("\n=== Analogy Test Set (Window 10) ===")
    print(f"Coverage: {analogy_results_window_10['coverage']}/{analogy_results_window_10['total']}")
    print(f"Top-5 accuracy: {analogy_results_window_10['accuracy_top5']:.4f}")
    for item in analogy_results_window_10["details"]:
        print(json.dumps(item, ensure_ascii=False))

    print("\n=== Direct Similarity Checks (Window 10) ===")
    for word1, word2 in check_pairs:
        if has_word(model_window_10, word1) and has_word(model_window_10, word2):
            similarity = cosine_similarity_score(model_window_10, word1, word2)
            print(f"{word1:10s} <-> {word2:10s}: {similarity:.4f}")
        else:
            print(f"{word1:10s} <-> {word2:10s}: OOV")

    print("\n=== Window Size Comparison ===")
    print(f"Window {WINDOW_SIZE} relatedness coverage: {relatedness_results['coverage']}/{relatedness_results['total']}")
    print(f"Window {WINDOW_SIZE_COMPARISON} relatedness coverage: {relatedness_results_window_10['coverage']}/{relatedness_results_window_10['total']}")
    print(f"Window {WINDOW_SIZE} analogy accuracy: {analogy_results['accuracy_top5']:.4f}")
    print(f"Window {WINDOW_SIZE_COMPARISON} analogy accuracy: {analogy_results_window_10['accuracy_top5']:.4f}")

    # ========================================================================
    # Model Export
    # ========================================================================
    
    model_path = "BSCS 3B\\LEAN_VINCE_CABALES\\exercise_5_skipgram_sgns.model"
    model.save(model_path)
    print(f"\nSaved model to: {model_path}")

    alt_model_path = "BSCS 3B\\LEAN_VINCE_CABALES\\exercise_5_skipgram_sgns_window_10.model"
    model_window_10.save(alt_model_path)
    print(f"Saved window-10 model to: {alt_model_path}")

    print("\nDone.")


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    main()
