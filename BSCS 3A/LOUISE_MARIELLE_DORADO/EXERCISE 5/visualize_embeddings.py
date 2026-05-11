"""
Visualize Word2Vec embeddings using PCA reduction to 2D.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from gensim.models import Word2Vec

# Load the trained model
model = Word2Vec.load("c:/Users/Marielle/Downloads/exercise_5_skipgram_sgns.model")

# Select at least 20 known words from the Bigfoot corpus
words = [
    "bigfoot", "sasquatch", "cryptid", "sighting", "forest",
    "footprint", "report", "evidence", "legend", "hoax",
    "investigating", "believers", "animal", "episode", "worth",
    "rockies", "investigations", "latest", "flick", "meets",
    "refuge", "sunset", "derry", "borough", "establishing"
]

# Extract vectors for these words
vectors = []
valid_words = []

for word in words:
    if word in model.wv.key_to_index:
        vectors.append(model.wv[word])
        valid_words.append(word)

vectors = np.array(vectors)

# Apply PCA to reduce to 2D
pca = PCA(n_components=2)
vectors_2d = pca.fit_transform(vectors)

# Create the plot
plt.figure(figsize=(14, 10))
plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], s=100, alpha=0.6, color='steelblue')

# Label each point with the word
for i, word in enumerate(valid_words):
    plt.annotate(word, (vectors_2d[i, 0], vectors_2d[i, 1]),
                 fontsize=10, ha='center', va='bottom',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))

plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
plt.title('Word2Vec Embeddings Visualization (PCA 100D → 2D)\nBigfoot Wikipedia Corpus (window=10, vector_size=100)', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save the figure
plt.savefig('c:/Users/Marielle/Downloads/word_embeddings_pca.png', dpi=150, bbox_inches='tight')
print(f"Visualization saved to: word_embeddings_pca.png")
print(f"Words visualized: {len(valid_words)}/{len(words)}")
print(f"PCA explained variance: PC1={pca.explained_variance_ratio_[0]:.1%}, PC2={pca.explained_variance_ratio_[1]:.1%}")
print(f"Total variance captured by 2D: {sum(pca.explained_variance_ratio_):.1%}")

plt.show()
