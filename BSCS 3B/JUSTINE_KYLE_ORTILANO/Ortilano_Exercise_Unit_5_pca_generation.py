
import re
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
from gensim.models import Word2Vec

# ── config ────────────────────────────────────────────────────────────────────
RANDOM_SEED    = 42
MODEL_PATH     = "exercise_5_skipgram_sgns.model"   # saved by exercise_5_minecraft.py
RUN_TRAINING   = False   # set True to re-train here instead of loading the saved model
WIKI_URL       = "https://en.wikipedia.org/wiki/Minecraft"
OUTPUT_IMAGE   = "exercise_5_pca.png"

# At least 20 words from the Minecraft article, grouped by semantic category
WORD_GROUPS = {
    "Gameplay / mechanics": [
        "minecraft", "sandbox", "survival", "crafting",
        "blocks", "gameplay", "mods", "multiplayer",
    ],
    "Mobs / creatures": [
        "creeper", "zombie", "mob", "skeleton", "enderman",
    ],
    "Platforms / editions": [
        "xbox", "playstation", "java", "bedrock", "release",
    ],
    "People / organizations": [
        "notch", "mojang", "microsoft", "developer",
    ],
    "Unrelated (control)": [
        "tractor", "kitchen", "astronomy",
    ],
}

# Color for each group
GROUP_COLORS = {
    "Gameplay / mechanics":   "#2a78d6",
    "Mobs / creatures":       "#1baf7a",
    "Platforms / editions":   "#eda100",
    "People / organizations": "#4a3aa7",
    "Unrelated (control)":    "#e34948",
}
# ──────────────────────────────────────────────────────────────────────────────


def maybe_train():
    """Optionally re-train the model instead of loading a saved one."""
    import requests
    from bs4 import BeautifulSoup
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize

    for r in ["punkt", "punkt_tab"]:
        try:
            nltk.data.find(f"tokenizers/{r}")
        except LookupError:
            nltk.download(r)

    print("Fetching Wikipedia article …")
    headers = {"User-Agent": "Mozilla/5.0 (SGNS-PCA/1.0)"}
    resp = requests.get(WIKI_URL, headers=headers, timeout=30)
    soup = BeautifulSoup(resp.text, "html.parser")
    content = soup.find("div", {"id": "mw-content-text"})
    paragraphs = content.find_all(["p", "li"])
    text = " ".join(p.get_text(" ", strip=True) for p in paragraphs)
    text = re.sub(r"\[[0-9]+\]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    sentences = sent_tokenize(text)
    processed = []
    for sent in sentences:
        sent = sent.lower()
        sent = re.sub(r"[^a-z0-9\-\s]", " ", sent)
        tokens = [t.strip("-") for t in word_tokenize(sent)]
        cleaned = [t for t in tokens if t and not t.isdigit() and len(t) >= 2]
        if len(cleaned) >= 3:
            processed.append(cleaned)

    print(f"Training on {len(processed)} sentences …")
    model = Word2Vec(
        sentences=processed,
        vector_size=100,
        window=10,
        min_count=1,
        workers=4,
        sg=1,
        negative=10,
        epochs=200,
        sample=1e-3,
        alpha=0.025,
        min_alpha=0.0007,
        seed=RANDOM_SEED,
    )
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    return model


def load_model():
    if RUN_TRAINING:
        return maybe_train()
    print(f"Loading model from {MODEL_PATH} …")
    return Word2Vec.load(MODEL_PATH)


def collect_vectors(model):
    """Return parallel lists: labels, vectors, colors."""
    labels, vectors, colors = [], [], []
    for group, words in WORD_GROUPS.items():
        color = GROUP_COLORS[group]
        for word in words:
            if word in model.wv.key_to_index:
                labels.append(word)
                vectors.append(model.wv[word])
                colors.append(color)
            else:
                print(f"  [OOV] '{word}' skipped")
    return labels, np.array(vectors), colors


def plot_pca(labels, vectors, colors):
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # ── PCA projection ────────────────────────────────────────────────────────
    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    coords = pca.fit_transform(vectors)
    explained = pca.explained_variance_ratio_ * 100

    # ── figure style ──────────────────────────────────────────────────────────
    plt.rcParams.update({
        "font.family":      "DejaVu Sans",
        "axes.spines.top":  False,
        "axes.spines.right":False,
        "axes.grid":        True,
        "grid.color":       "#e1e0d9",
        "grid.linewidth":   0.6,
        "figure.facecolor": "#ffffff",
        "axes.facecolor":   "#ffffff",
    })

    fig, ax = plt.subplots(figsize=(10, 7))

    # ── scatter points ────────────────────────────────────────────────────────
    ax.scatter(
        coords[:, 0], coords[:, 1],
        c=colors,
        s=90,
        zorder=3,
        linewidths=0.8,
        edgecolors="white",
    )

    # ── word labels with simple collision nudge ───────────────────────────────
    for i, (label, (x, y)) in enumerate(zip(labels, coords)):
        offset_x = 0.1
        # increase vertical offset so text appears more above the point
        offset_y = 0.04
        # nudge a few words that overlap each other (place them below)
        if label in {"sandbox", "crafting", "gameplay"}:
            offset_y = -0.03
        ax.text(
            x + offset_x, y + offset_y,
            label,
            fontsize=9,
            color=colors[i],
            zorder=4,
        )

    # ── axes labels ───────────────────────────────────────────────────────────
    ax.set_xlabel(f"PC 1 ({explained[0]:.1f}% variance)", fontsize=11, color="#52514e")
    ax.set_ylabel(f"PC 2 ({explained[1]:.1f}% variance)", fontsize=11, color="#52514e")
    ax.tick_params(colors="#898781", labelsize=9)

    # ── title ─────────────────────────────────────────────────────────────────
    ax.set_title(
        "PCA of Word2Vec embeddings — Minecraft Wikipedia",
        fontsize=13,
        fontweight="medium",
        color="#0b0b0b",
        pad=14,
    )

    # ── legend ────────────────────────────────────────────────────────────────
    patches = [
        mpatches.Patch(color=color, label=group)
        for group, color in GROUP_COLORS.items()
    ]
    ax.legend(
        handles=patches,
        fontsize=9,
        frameon=True,
        framealpha=0.9,
        edgecolor="#e1e0d9",
        loc="lower left",
    )

    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to: {OUTPUT_IMAGE}")
    plt.show()


def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    model  = load_model()
    labels, vectors, colors = collect_vectors(model)

    print(f"\n{len(labels)} words found in vocabulary:")
    for lbl in labels:
        print(f"  {lbl}")

    if len(labels) < 2:
        print("Need at least 2 in-vocabulary words to plot. Exiting.")
        return

    plot_pca(labels, vectors, colors)


main()
