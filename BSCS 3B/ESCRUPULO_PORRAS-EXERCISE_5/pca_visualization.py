from gensim.models import Word2Vec
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load your trained model
model = Word2Vec.load("model.model")

# words = list(model.wv.index_to_key)   # Checking the the words available in Model
# print(words[:40])  # show first 20 words

# Selected words 
words = [
    'philippine', 'filipino', 'history', 'spain', 'spanish',
    'manila', 'war', 'state', 'country', 'island',
    'asia', 'united', 'archive', 'original', 'retrieve',
    'new', 'which', 'their', 'have', 'also'
]

# Filter words that actually exist in the model (VERY IMPORTANT)
words = [word for word in words if word in model.wv]

# Get vectors
vectors = [model.wv[word] for word in words]

# Apply PCA
pca = PCA(n_components=2)
result = pca.fit_transform(vectors)

# Plot
plt.figure(figsize=(10, 8))
plt.scatter(result[:, 0], result[:, 1])

# Add labels
for i, word in enumerate(words):
    plt.annotate(word, xy=(result[i, 0], result[i, 1]))

plt.title("Word2Vec PCA Visualization (Philippine History Words)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.grid()

plt.show()


