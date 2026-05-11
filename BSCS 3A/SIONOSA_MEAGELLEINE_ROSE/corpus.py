import numpy as np

corpus = [
    "Climate change affects global weather patterns significantly".split(),
    "Rising sea levels threaten coastal cities worldwide".split(),
    "Deforestation contributes to higher carbon dioxide levels".split(),
    "Renewable energy sources like solar and wind are essential".split(),
    "Solar panels convert sunlight into clean electricity".split(),
    "Wind turbines generate power from moving air currents".split(),
    "Greenhouse gases trap heat in Earth's atmosphere".split(),
    "Carbon emissions from factories pollute the air quality".split(),
    "Electric vehicles reduce dependence on fossil fuels".split(),
    "Fossil fuels such as coal and oil are finite resources".split(),
    "Biodiversity loss impacts ecosystems and food chains".split(),
    "Conservation efforts protect endangered species habitats".split(),
    "Recycling reduces waste and conserves natural resources".split(),
    "Sustainable agriculture promotes soil health and water conservation".split(),
    "International agreements aim to limit global warming".split(),
]

# Lowercase all words
corpus = [[w.lower() for w in sent] for sent in corpus]

# Build vocabulary
vocab = sorted({w for sent in corpus for w in sent})
word_to_id = {w: i for i, w in enumerate(vocab)}
V = len(vocab)

print(f"Vocabulary size: {V}")
print(f"Vocabulary: {vocab}\n")

# Build co-occurrence matrix with ±3 window
M = np.zeros((V, V), dtype=int)
window_size = 3

for sent in corpus:
    n = len(sent)
    for i, w in enumerate(sent):
        w_id = word_to_id[w]
        for j in range(max(0, i - window_size), min(n, i + window_size + 1)):
            if j == i:
                continue
            c_id = word_to_id[sent[j]]
            M[w_id, c_id] += 1

print("Co-occurrence Matrix:")
print("Shape:", M.shape)
print(M)
print()

# Compute L2 vector lengths
vector_lengths = np.sqrt(np.sum(M**2, axis=1))

# Top 10 longest vectors
top10_idx = np.argsort(vector_lengths)[::-1][:10]
print("Top 10 Longest Vectors (by L2 norm):")
print(f"{'Rank':<5} {'Word':<25} {'L2 Norm':<12} Non-zero Co-occurrences")
print("-" * 100)
for rank, idx in enumerate(top10_idx, 1):
    word = vocab[idx]
    norm = vector_lengths[idx]
    vec = M[idx]
    nonzero = [(vocab[j], int(vec[j])) for j in range(V) if vec[j] > 0]
    print(f"{rank:<5} {word:<25} {norm:<12.4f} {nonzero}")