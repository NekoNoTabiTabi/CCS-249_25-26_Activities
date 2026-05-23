"""
Task 5: Visualize Word Embeddings using PCA
Reduce 100-dimensional vectors to 2D using PCA and visualize
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from gensim.models import Word2Vec


def visualize_word_embeddings(model: Word2Vec, words: list, title: str = "Word Embeddings Visualization (PCA)"):
    """
    Visualize word embeddings in 2D space using PCA.
    
    Args:
        model: Trained Word2Vec model
        words: List of words to visualize
        title: Title for the visualization
    """
    # Filter words that exist in vocabulary
    valid_words = [w for w in words if w in model.wv.key_to_index]
    print(f"Visualizing {len(valid_words)}/{len(words)} words\n")
    
    if len(valid_words) < 2:
        print("Not enough valid words to visualize!")
        return
    
    # Get word vectors
    vectors = np.array([model.wv[w] for w in valid_words])
    
    # Apply PCA to reduce to 2D
    pca = PCA(n_components=2)
    vectors_2d = pca.fit_transform(vectors)
    
    print(f"PCA Explained Variance Ratio: {pca.explained_variance_ratio_}")
    print(f"Total Variance Explained: {sum(pca.explained_variance_ratio_):.2%}\n")
    
    # Create figure
    plt.figure(figsize=(14, 10))
    
    # Plot points
    plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], s=100, alpha=0.6, c='steelblue', edgecolors='navy', linewidth=1.5)
    
    # Add word labels
    for i, word in enumerate(valid_words):
        plt.annotate(
            word,
            xy=(vectors_2d[i, 0], vectors_2d[i, 1]),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=10,
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', lw=0.5)
        )
    
    # Formatting
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=12, fontweight='bold')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    # Save figure
    plt.savefig('word_embeddings_pca_visualization.png', dpi=300, bbox_inches='tight')
    print("✓ Visualization saved as: word_embeddings_pca_visualization.png\n")
    
    plt.show()


if __name__ == "__main__":
    print("=== TASK 5: WORD EMBEDDINGS VISUALIZATION (PCA) ===\n")
    
    # Load trained model
    print("Loading trained Word2Vec model...")
    try:
        model = Word2Vec.load("skipgram_model.model")
        print(f"✓ Model loaded successfully")
        print(f"  Vocabulary size: {len(model.wv)}")
        print(f"  Vector dimension: {model.wv.vector_size}\n")
    except FileNotFoundError:
        print("✗ Error: skipgram_model.model not found!")
        print("  Please run task_2_training.py first")
        exit(1)
    
    # Define probe words (JoJo's related + general terms)
    probe_words = [
        # Main JoJo's terms
        "jojo", "bizarre", "stand", "power", "adventure",
        "anime", "series", "character", "episode", "manga",
        "protagonist", "villain", "antagonist", "story", "plot",
        "arc", "battle", "ability", "strength", "ability",
        "fighting", "supernatural", "mystery", "drama", "action"
    ]
    
    # Filter unique words
    probe_words = list(dict.fromkeys(probe_words))
    
    print(f"Probing with {len(probe_words)} unique words:\n")
    for i, word in enumerate(probe_words, 1):
        status = "✓" if word in model.wv.key_to_index else "✗"
        print(f"  {i:2d}. {status} {word:20s}")
    
    print("\n" + "="*70 + "\n")
    
    # Visualize
    visualize_word_embeddings(
        model, 
        probe_words, 
        title="Skip-Gram Word Embeddings (Window=10, Vector=100) - PCA Projection"
    )
    
    print("\n[INTERPRETATION]")
    print("• Words positioned close together have similar semantic meanings")
    print("• Words far apart have different contextual relationships")
    print("• Clusters indicate thematic groups (e.g., 'jojo', 'bizarre', 'adventure' cluster)")
    print("• PC1 and PC2 axes represent the two principal components of variance")
