"""
Task 2 (10 points): Train a Skip-gram with Negative Sampling Model
Train Word2Vec using the skip-gram architecture with negative sampling.

This task involves:
- Configuring the model hyperparameters (vector_size, window, negative sampling)
- Training the model on the corpus
- Noting key properties that affect the quality of learned embeddings
"""

import pickle
import random
import numpy as np
from typing import List
from gensim.models import Word2Vec

# Configuration: Key hyperparameters for Skip-gram training
# TASK 2b: These are properties to note and can be experimented with

VECTOR_SIZE = 100       # Dimensionality of word vectors
                        # Try: 50, 100, 200, 300
                        # Larger vectors capture more nuanced relationships but need more data

WINDOW_SIZE = 10        # Context window size (words on each side)
                        # Try: 3, 5, 8, 10
                        # Larger window captures broader semantic relationships

NEGATIVE_SAMPLES = 10   # Number of negative samples in negative sampling
                        # Try: 5, 10, 15, 20
                        # More samples = cleaner gradient updates

EPOCHS = 200           # Number of training iterations
WORKERS = 4            # Number of worker threads
MIN_COUNT = 1          # Minimum word frequency to include in vocabulary
ALPHA = 0.025          # Initial learning rate
MIN_ALPHA = 0.0007     # Final learning rate
SAMPLE = 1e-3          # Threshold for subsampling frequent words
RANDOM_SEED = 42


def train_skipgram_negative_sampling(sentences: List[List[str]]) -> Word2Vec:
    """
    TASK 2a: Train a Skip-gram with Negative Sampling model.
    
    This uses gensim's Word2Vec with:
    - sg=1: Skip-gram (not CBOW)
    - negative=NEGATIVE_SAMPLES: Use negative sampling (not hierarchical softmax)
    
    Args:
        sentences: List of tokenized sentences from the corpus
        
    Returns:
        Trained Word2Vec model
    """
    print("=== TASK 2: Training Skip-gram with Negative Sampling ===\n")
    
    print(f"Model Configuration:")
    print(f"  Vector Size:       {VECTOR_SIZE}")
    print(f"  Window Size:       {WINDOW_SIZE}")
    print(f"  Negative Samples:  {NEGATIVE_SAMPLES}")
    print(f"  Epochs:            {EPOCHS}")
    print()
    
    model = Word2Vec(
        sentences=sentences,
        vector_size=VECTOR_SIZE,      # TASK 2b: Dimensionality of embeddings
        window=WINDOW_SIZE,            # TASK 2b: Context window
        min_count=MIN_COUNT,
        workers=WORKERS,
        sg=1,                          # 1 = skip-gram, 0 = CBOW
        negative=NEGATIVE_SAMPLES,     # Use negative sampling and number of negative samples
        epochs=EPOCHS,
        sample=SAMPLE,
        alpha=ALPHA,
        min_alpha=MIN_ALPHA,
        seed=RANDOM_SEED,
    )
    
    return model


if __name__ == "__main__":
    print("Loading preprocessed corpus...")
    with open("corpus_sentences.pkl", "rb") as f:
        sentences = pickle.load(f)
    
    print(f"Loaded {len(sentences)} sentences\n")
    
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    # Train the model
    model = train_skipgram_negative_sampling(sentences)
    
    print(f"Training complete!")
    print(f"Vocabulary size: {len(model.wv)} words\n")
    
    # TASK 2c: Code line identification
    # Lines related to Skip-gram training with Negative Sampling:
    # - Line with "sg=1": Specifies skip-gram architecture
    # - Line with "negative=NEGATIVE_SAMPLES": Enables negative sampling
    # - Line with "Word2Vec(": Start of model instantiation (lines ~47-59)
    
    # Save model for use in evaluation
    model.save("skipgram_model.model")
    print("Saved model to: skipgram_model.model")
