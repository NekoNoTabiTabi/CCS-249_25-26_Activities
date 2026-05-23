from sklearn.metrics.pairwise import cosine_similarity

def cosine(model, w1, w2):
    return model.wv.similarity(w1, w2)