# Import necessary libraries
import wikipediaapi
import re
import nltk
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def fetch_wikipedia_content(topic):
    wiki = wikipediaapi.Wikipedia('Enryu (dukesalfredbocala4@gmail.com)', 'en')
    page = wiki.page(topic)
    if page.exists():
        return page.text
    return ""

corpus_text = fetch_wikipedia_content("Artificial intelligence")
print(f"Fetched {len(corpus_text.split())} words from Wikipedia.")

def preprocess_text(text):
    # Remove non-alphabetic characters and lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text).lower()
    tokens = text.split()
    # Remove stopwords
    tokens = [w for w in tokens if w not in stop_words and len(w) > 2]
    return tokens

processed_tokens = preprocess_text(corpus_text)
sentences = [processed_tokens] # Word2Vec expects a list of lists

# --- (10 points) Train a Skip-gram with Negative Sampling model ---
# Codeline: Word2Vec(sentences, sg=1, negative=5, ...)
# Initial configuration: Window=5, Vector Size=100
model_v1 = Word2Vec(
    sentences=sentences, 
    vector_size=100, 
    window=5, 
    min_count=2, 
    sg=1,          # 1 for Skip-gram
    negative=5,    # Negative sampling
    workers=4
)

test_words = ["intelligence", "machine", "learning", "data", "robot"]

def evaluate_model(model, words):
    print(f"\nEvaluation for Window Size {model.window}:")
    for word in words:
        if word in model.wv:
            neighbors = model.wv.most_similar(word, topn=3)
            print(f"Word: {word} | Neighbors: {neighbors}")

evaluate_model(model_v1, test_words)


model_v2 = Word2Vec(
    sentences=sentences, 
    vector_size=100, 
    window=10, 
    min_count=2, 
    sg=1, 
    negative=5
)
evaluate_model(model_v2, test_words)

def visualize_embeddings(model, words):
    word_vectors = [model.wv[w] for w in words if w in model.wv]
    valid_words = [w for w in words if w in model.wv]
    
    pca = PCA(n_components=2)
    result = pca.fit_transform(word_vectors)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(result[:, 0], result[:, 1])
    for i, word in enumerate(valid_words):
        plt.annotate(word, xy=(result[i, 0], result[i, 1]))
    plt.title("PCA Visualization of Word Embeddings")
    plt.show()


viz_words = ["intelligence", "machine", "learning", "data", "robot", "computer", 
             "human", "system", "algorithm", "software", "network", "science", 
             "technology", "research", "process", "digital", "brain", "model", 
             "logic", "future"]
visualize_embeddings(model_v2, viz_words)