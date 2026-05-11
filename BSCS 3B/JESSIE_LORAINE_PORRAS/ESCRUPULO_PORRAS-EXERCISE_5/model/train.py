from gensim.models import Word2Vec
from config import *

def train_sgns(sentences):
    return Word2Vec(
        sentences=sentences,
        vector_size= VECTOR_SIZE,
        window=WINDOW,
        min_count=1,
        workers=4,
        sg=1,
        negative=NEGATIVE,
        epochs=EPOCHS,
        seed=RANDOM_SEED,
    )