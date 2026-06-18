"""
# Generated using AI

import numpy as np

# This code calculates the co-occurrence matrix for a given corpus of documents. 
# The co-occurrence matrix counts how many times each pair of words appears together in the same document.

# Tokenized corpus
# You can replace this with your own text
corpus = [
    ["apple", "banana", "apple", "orange"],
    ["banana", "orange", "banana", "apple"],
    ["grape", "banana", "apple", "banana"],
    ["orange", "grape", "banana", "apple"],

    #["Climate change affects global weather patterns significantly."],
    #["Rising sea levels threaten coastal cities worldwide."],
    #["Deforestation contributes to higher carbon dioxide levels."],
    #["Renewable energy sources like solar and wind are essential."],
    #["Solar panels convert sunlight into clean electricity."],
    #["Wind turbines generate power from moving air currents."],
    #["Greenhouse gases trap heat in Earth's atmosphere."],
    #["Carbon emissions from factories pollute the air quality."],
    #["Electric vehicles reduce dependence on fossil fuels."],
    #["Fossil fuels such as coal and oil are finite resources."],
    #["Biodiversity loss impacts ecosystems and food chains."],
    #["Conservation efforts protect endangered species habitats."],
    #["Recycling reduces waste and conserves natural resources."],
    #["Sustainable agriculture promotes soil health and water conservation."],
    ##["International agreements aim to limit global warming."],

]

# Since we are interested in co-occurrence, we will create a vocabulary of unique words
# Build vocabulary
vocab = sorted({w for sent in corpus for w in sent})
word_to_id = {w: i for i, w in enumerate(vocab)}
V = len(vocab)
M = np.zeros((V, V), dtype=int)

window_size = 1  # one word to left/right

for sent in corpus:
    n = len(sent)
    for i, w in enumerate(sent):
        w_id = word_to_id[w]
        # context positions
        for j in range(max(0, i - window_size), min(n, i + window_size + 1)):
            if j == i:
                continue
            c = sent[j]
            c_id = word_to_id[c]
            M[w_id, c_id] += 1
            M[c_id, w_id] += 0  # keep symmetric if you want; or count once per (center,context)

print("Vocab:", vocab)
print("Co-occurrence matrix:\n", M)
"""


import numpy as np
import string

# 1. Prepare the corpus from the assignment
raw_corpus = [
    "Climate change affects global weather patterns significantly.",
    "Rising sea levels threaten coastal cities worldwide.",
    "Deforestation contributes to higher carbon dioxide levels.",
    "Renewable energy sources like solar and wind are essential.",
    "Solar panels convert sunlight into clean electricity.",
    "Wind turbines generate power from moving air currents.",
    "Greenhouse gases trap heat in Earth's atmosphere.",
    "Carbon emissions from factories pollute the air quality.",
    "Electric vehicles reduce dependence on fossil fuels.",
    "Fossil fuels such as coal and oil are finite resources.",
    "Biodiversity loss impacts ecosystems and food chains.",
    "Conservation efforts protect endangered species habitats.",
    "Recycling reduces waste and conserves natural resources.",
    "Sustainable agriculture promotes soil health and water conservation.",
    "International agreements aim to limit global warming."
]

# Tokenization: lowercase and remove punctuation
corpus = [sent.lower().translate(str.maketrans('', '', string.punctuation)).split() for sent in raw_corpus]

# Build vocabulary
vocab = sorted({w for sent in corpus for w in sent})
word_to_id = {w: i for i, w in enumerate(vocab)}
id_to_word = {i: w for i, w in enumerate(vocab)}
V = len(vocab)
M = np.zeros((V, V), dtype=int)

# Set context window of +-3 words
window_size = 3  

# 2. Build the Co-occurrence Matrix
for sent in corpus:
    n = len(sent)
    for i, w in enumerate(sent):
        w_id = word_to_id[w]
        for j in range(max(0, i - window_size), min(n, i + window_size + 1)):
            if j == i:
                continue
            c_id = word_to_id[sent[j]]
            M[w_id, c_id] += 1

# 3. Calculate Magnitudes for the Top 10 Longest Vectors
magnitudes = np.linalg.norm(M, axis=1)
top_10_idx = np.argsort(magnitudes)[-10:][::-1]

print("--- Top 10 Longest Vectors (Ranked by Magnitude) ---")
for rank, idx in enumerate(top_10_idx, 1):
    word = id_to_word[idx]
    mag = magnitudes[idx]
    print(f"{rank}. '{word}' | Magnitude: {mag:.4f}")

# 4. Cosine Similarity Function
def cosine_sim(w1, w2):
    v1, v2 = M[word_to_id[w1]], M[word_to_id[w2]]
    norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
    return np.dot(v1, v2) / (norm1 * norm2) if norm1 and norm2 else 0.0

print("\n--- Cosine Similarities ---")
pairs = [("climate", "global"), ("solar", "wind"), ("fossil", "coal")]
for w1, w2 in pairs:
    print(f"Similarity({w1}, {w2}) = {cosine_sim(w1, w2):.4f}")