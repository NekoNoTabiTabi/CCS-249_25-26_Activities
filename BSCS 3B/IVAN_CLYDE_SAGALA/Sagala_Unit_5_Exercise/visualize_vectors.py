"""Visualize trained Word2Vec vectors with PCA.

Usage examples:
  python visualize_vectors.py                    # uses exercise_5_skipgram_sgns.model and top 20 words
  python visualize_vectors.py --model my.model --top 30 --out my_viz.png
  python visualize_vectors.py --words car,engine,truck,road --out sel.png
"""

import sys
import argparse
from gensim.models import Word2Vec
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def load_model(path: str) -> Word2Vec:
    return Word2Vec.load(path)


def get_top_words(model: Word2Vec, top: int):
    return list(model.wv.index_to_key)[:top]


def plot_words(words, vectors, out_path: str):
    pca = PCA(n_components=2)
    coords = pca.fit_transform(np.array(vectors))

    plt.figure(figsize=(12, 9))
    xs = coords[:, 0]
    ys = coords[:, 1]
    plt.scatter(xs, ys, s=40, alpha=0.7)

    for x, y, w in zip(xs, ys, words):
        plt.text(x + 0.01, y + 0.01, w, fontsize=9)

    plt.title(f"PCA projection of {len(words)} word vectors")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved visualization to: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="exercise_5_skipgram_sgns.model", help="Path to saved Word2Vec model")
    parser.add_argument("--top", type=int, default=20, help="Number of top-frequency words to plot (ignored if --words provided)")
    parser.add_argument("--words", type=str, help="Comma-separated words to plot (overrides --top)")
    parser.add_argument("--out", default="vector_pca.png", help="Output image path")
    args = parser.parse_args()

    try:
        model = load_model(args.model)
    except Exception as e:
        print("Error loading model:", e)
        sys.exit(1)

    if args.words:
        requested = [w.strip() for w in args.words.split(",") if w.strip()]
    else:
        requested = get_top_words(model, args.top)

    present = []
    vectors = []
    for w in requested:
        if w in model.wv.key_to_index:
            present.append(w)
            vectors.append(model.wv[w])
        else:
            print(f"Warning: '{w}' not in model vocabulary; skipping")

    if not vectors:
        print("No valid words to visualize. Exiting.")
        sys.exit(1)

    plot_words(present, vectors, args.out)


if __name__ == '__main__':
    main()
