import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load the fine-tuned model and tokenizer
model_path = "./emotion-distilbert-final"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Define at least 20 known words
words = [
    "happy", "sad", "angry", "fear", "love", "surprise",
    "joy", "hate", "excited", "calm", "worried", "peaceful",
    "furious", "delighted", "anxious", "content", "frustrated", "amazed",
    "disgusted", "grateful"
]

# Function to get embeddings for a word
def get_word_embedding(word):
    inputs = tokenizer(word, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        # Get the last hidden state (embeddings from the last layer)
        hidden_states = outputs.hidden_states[-1]  # Shape: (batch_size, seq_len, hidden_size)
        # Average the token embeddings (excluding special tokens if any)
        # For single words, this will be the embedding of the word tokens
        embedding = hidden_states.mean(dim=1).squeeze().numpy()  # Shape: (hidden_size,)
    return embedding

# Get embeddings for all words
embeddings = []
for word in words:
    emb = get_word_embedding(word)
    embeddings.append(emb)

embeddings = np.array(embeddings)  # Shape: (num_words, hidden_size)

# Apply PCA to reduce to 2D
pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(embeddings)

# Plot the results
plt.figure(figsize=(12, 8))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c='blue', alpha=0.7)

# Add labels
for i, word in enumerate(words):
    plt.annotate(word, (embeddings_2d[i, 0], embeddings_2d[i, 1]), fontsize=10, alpha=0.8)

plt.title("PCA Visualization of Word Embeddings from Fine-tuned DistilBERT")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save the plot
plt.savefig("word_embeddings_pca.png", dpi=300, bbox_inches='tight')
plt.show()

print("PCA visualization saved as 'word_embeddings_pca.png'")
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
print(f"Total explained variance: {np.sum(pca.explained_variance_ratio_):.4f}")