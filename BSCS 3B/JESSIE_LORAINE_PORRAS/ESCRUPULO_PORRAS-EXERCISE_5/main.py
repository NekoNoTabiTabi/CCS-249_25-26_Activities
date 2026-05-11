from pyexpat import model

from config import *
from data.fetch import fetch_wikipedia_article
from data.preprocess import preprocess_text
from model.train import train_sgns
from evaluation.analogy import evaluate_analogies
from evaluation.similarity import cosine


def main():

    # List of analogies to test after training the model. 
    test_analogies = [
        ('spain', 'spanish', 'america', 'american'),
        ('manila', 'luzon', 'cebu', 'visayas'),
        ('war', 'spanish', 'occupation', 'japanese'), 
        ('colonization', 'spain', 'administration', 'america'),
        ('president', 'republic', 'emperor', 'japan'),
        ('manila', 'philippines', 'tokyo', 'japan'),
        ('century', 'centuries', 'year', 'years'),
        ('magellan', 'portugal', 'legazpi', 'spain'),
        ('island', 'islands', 'nation', 'nations'),
        ('independence', '1946', 'revolution', '1896'),
    ]

    # data pipeline
    print("Fetching article...")
    text = fetch_wikipedia_article(WIKI_URL)
    print("Preprocessing...")
    sentences = preprocess_text(text)

    # Training the model
    print("Training model...")
    model = train_sgns(sentences)
    print("Done! Vocabulary size:", len(model.wv))
    model.save("model.model")

    # Evaluation 
    print("\n" + "="*30)
    print("REPORT: NEAREST NEIGHBORS")
    print("="*30)
    
    for word in ['spain', 'rizal', 'war']:
        if word in model.wv:
            neighbors = model.wv.most_similar(word, topn=5)
            print(f"{word.upper()}: {neighbors}")

    print("\n" + "="*30)
    print("REPORT: SIMILARITY SCORES")
    print("\n" + "="*30)
    
    pairs = [('spain', 'america'), ('manila', 'cebu'), ('war', 'occupation')]
    for w1, w2 in pairs:
        if w1 in model.wv and w2 in model.wv:
            score = cosine(model, w1, w2)
            print(f"Cosine similarity between '{w1}' and '{w2}': {score:.4f}")

    print("\n" + "="*30)
    print("REPORT: ANALOGY EVALUATION (TEST-SET PERFORMANCE)")
    print("\n" + "="*30)

    results = evaluate_analogies(model, test_analogies)
    for a, b, c, d, preds in results:
        status = "CORRECT" if d in preds else "INCORRECT"
        print(f"{a}:{b} :: {c}:{d} -> {status} | Predicted: {preds}")

if __name__ == "__main__":
    main()

  