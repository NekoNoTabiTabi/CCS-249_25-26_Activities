import re
from collections import defaultdict


# DATASET

dataset = [
    ("Free money now!!!",                       "SPAM"),
    ("Hi mom, how are you?",                    "HAM"),
    ("Lowest price for your meds",              "SPAM"),
    ("Are we still on for dinner?",             "HAM"),
    ("Win a free iPhone today",                 "SPAM"),
    ("Let's catch up tomorrow at the office",   "HAM"),
    ("Meeting at 3 PM tomorrow",                "HAM"),
    ("Get 50% off, limited time!",              "SPAM"),
    ("Team meeting in the office",              "HAM"),
    ("Click here for prizes!",                  "SPAM"),
    ("Can you send the report?",                "HAM"),
]


# HELPER: Tokeniser
def tokenize(text: str) -> list[str]:
    """Lowercase and extract alphabetic tokens only."""
    return re.findall(r"[a-z]+", text.lower())



# 1.) A. GENERATE BAG OF WORDS (for word frequency)

def generate_bag_of_words(data: list[tuple[str, str]]) -> dict:
    """
    Returns:
        bow  – {class_label: {word: count}}
    """
    bow = defaultdict(lambda: defaultdict(int))
    for text, label in data:
        for word in tokenize(text):
            bow[label][word] += 1
    return bow



# 1.) B. Calculate the prior for the class HAM and SPAM

def calculate_priors(data: list[tuple[str, str]]) -> dict:
    """
    Returns:
        priors – {class_label: probability}
    """
    class_counts = defaultdict(int)
    for _, label in data:
        class_counts[label] += 1
    total = len(data)
    priors = {label: count / total for label, count in class_counts.items()}
    return priors


# 1.) C. Calculate the likelihood of the tokens in the vocabulary with respect to the class.
# (with Laplace / add-1 smoothing)
#    P(word | class) = (count(word, class) + 1)
#                      / (total_words_in_class + vocab_size)

def calculate_likelihoods(
    bow: dict,
    vocabulary: set,
    alpha: int = 1     # Laplace smoothing parameter
) -> dict:
    """
    Returns:
        likelihoods – {class_label: {word: P(word|class)}}
    """
    likelihoods = {}
    vocab_size = len(vocabulary)

    for label, word_counts in bow.items():
        total_words = sum(word_counts.values())
        likelihoods[label] = {}
        for word in vocabulary:
            count = word_counts.get(word, 0)
            likelihoods[label][word] = (count + alpha) / (total_words + vocab_size * alpha)

    return likelihoods



# 1.) D. Determine the class of the following test sentence:  i. Limited offer, click here!   ii. Meeting at 2 PM with the manager.
#    score(class) = log P(class) + Σ log P(word | class)
#    (log-space to avoid underflow)

import math

def classify(
    text: str,
    priors: dict,
    likelihoods: dict,
    vocabulary: set,
    alpha: int = 1
) -> tuple[str, dict]:

    tokens = tokenize(text)
    scores = {}

    for label in priors:
        log_score = math.log(priors[label])
        total_words = sum(1 for w in likelihoods[label])   # |V|
        for token in tokens:
            if token in likelihoods[label]:
                log_score += math.log(likelihoods[label][token])
            # OOV tokens: already handled by Laplace — use smallest smoothed prob
            # (Here every token in the test set is in the vocabulary for this dataset)
        scores[label] = log_score

    predicted = max(scores, key=scores.get)
    return predicted, scores



# all steps and print results

if __name__ == "__main__":

    SEP  = "=" * 70
    SEP2 = "-" * 70

    # ── Step A: Bag of Words ──────────────────────────────────────────────
    bow = generate_bag_of_words(dataset)

    # Full vocabulary (union of both classes)
    vocabulary = set()
    for label in bow:
        vocabulary.update(bow[label].keys())

    print(SEP)
    print("  A. BAG OF WORDS (word frequency per class)")
    print(SEP)

    all_words_sorted = sorted(vocabulary)
    header = f"{'WORD':<20} {'HAM':>6} {'SPAM':>6}"
    print(header)
    print(SEP2)
    for word in all_words_sorted:
        h = bow["HAM"].get(word, 0)
        s = bow["SPAM"].get(word, 0)
        print(f"{word:<20} {h:>6} {s:>6}")

    print(SEP2)
    total_ham  = sum(bow["HAM"].values())
    total_spam = sum(bow["SPAM"].values())
    print(f"{'TOTAL TOKENS':<20} {total_ham:>6} {total_spam:>6}")
    print(f"Vocabulary size (|V|) = {len(vocabulary)}")

    # ── Step B: Priors ────────────────────────────────────────────────────
    priors = calculate_priors(dataset)

    n_total = len(dataset)
    n_ham   = sum(1 for _, l in dataset if l == "HAM")
    n_spam  = sum(1 for _, l in dataset if l == "SPAM")

    print()
    print(SEP)
    print("  B. PRIOR PROBABILITIES")
    print(SEP)
    print(f"  Total documents  : {n_total}")
    print(f"  HAM  documents   : {n_ham}")
    print(f"  SPAM documents   : {n_spam}")
    print()
    print(f"  P(HAM)  = {n_ham}/{n_total} = {priors['HAM']:.6f}")
    print(f"  P(SPAM) = {n_spam}/{n_total} = {priors['SPAM']:.6f}")

    # ── Step C: Likelihoods ───────────────────────────────────────────────
    likelihoods = calculate_likelihoods(bow, vocabulary, alpha=1)

    V = len(vocabulary)
    print()
    print(SEP)
    print("  C. LIKELIHOOD  P(word | class)  — Laplace (add-1) smoothing")
    print(f"     Formula: P(w|c) = (count(w,c) + 1) / (total_words_in_c + |V|)")
    print(f"     |V| = {V},  total HAM tokens = {total_ham},  total SPAM tokens = {total_spam}")
    print(SEP)
    print(f"{'WORD':<20} {'count|HAM':>10} {'P(w|HAM)':>12} {'count|SPAM':>11} {'P(w|SPAM)':>12}")
    print(SEP2)
    for word in all_words_sorted:
        ch = bow["HAM"].get(word, 0)
        cs = bow["SPAM"].get(word, 0)
        ph = likelihoods["HAM"][word]
        ps = likelihoods["SPAM"][word]
        print(f"{word:<20} {ch:>10} {ph:>12.6f} {cs:>11} {ps:>12.6f}")

    # ── Step D: Classification ────────────────────────────────────────────
    test_sentences = [
        "Limited offer, click here!",
        "Meeting at 2 PM with the manager.",
    ]

    print()
    print(SEP)
    print("  D. CLASSIFICATION OF TEST SENTENCES")
    print(SEP)

    for sentence in test_sentences:
        tokens = tokenize(sentence)
        predicted, scores = classify(sentence, priors, likelihoods, vocabulary)

        print(f"\n  Sentence : \"{sentence}\"")
        print(f"  Tokens   : {tokens}")
        print()

        # Show step-by-step computation for each class
        for label in ["HAM", "SPAM"]:
            log_prior = math.log(priors[label])
            print(f"  [{label}]")
            print(f"    log P({label}) = log({priors[label]:.6f}) = {log_prior:.6f}")
            running = log_prior
            for token in tokens:
                if token in likelihoods[label]:
                    lp = math.log(likelihoods[label][token])
                    print(f"    + log P('{token}'|{label}) "
                          f"= log({likelihoods[label][token]:.6f}) = {lp:.6f}")
                    running += lp
                else:
                    print(f"    + log P('{token}'|{label}) → OOV (skipped)")
            print(f"    ─────────────────────────────────")
            print(f"    Total log-score = {running:.6f}")
            print()

        print(f"Predicted Class : {predicted}")
        print(f"     (log-score HAM={scores['HAM']:.6f},  SPAM={scores['SPAM']:.6f})")
        print(SEP2)

    # 2.) Using Scikit-Learn
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.naive_bayes import MultinomialNB
    import numpy as np

    # --- Training ---
    X_texts = [text for text, _ in dataset]
    y_labels = [label for _, label in dataset]

    # CountVectorizer builds the vocabulary and produces the BoW matrix
    vectorizer = CountVectorizer(lowercase=True, token_pattern=r"[a-z]+")
    X_train = vectorizer.fit_transform(X_texts)

    model = MultinomialNB(alpha=1.0)
    model.fit(X_train, y_labels)

    print()
    print(SEP)
    print("  2. USING SCIKIT-LEARN (MultinomialNB)")
    print(SEP)

    # vocabulary learned by CountVectorizer
    vocab = vectorizer.get_feature_names_out()
    print(f"  Vocabulary size : {len(vocab)}")
    print(f"  Vocabulary      : {sorted(vocab)}")
    print()

    # training class distribution
    classes = model.classes_
    print(f"  Classes         : {[str(c) for c in classes]}")
    print(f"  Class log-priors:")
    for cls, lp in zip(classes, model.class_log_prior_):
        cls = str(cls)
        n = y_labels.count(cls)
        print(f"    log P({cls})  = log({n}/{len(dataset)}) = {lp:.6f}  →  P({cls}) = {math.exp(lp):.6f}")
    print()

    # --- Testing ---
    sk_test_sentences = [
        "Limited offer, click here!",
        "Meeting at 2 PM with the manager.",
    ]
    X_test = vectorizer.transform(sk_test_sentences)
    preds        = model.predict(X_test)
    log_probs    = model.predict_log_proba(X_test)   # log-posterior per class
    proba        = model.predict_proba(X_test)        # actual posterior probabilities

    print(f"  {'SENTENCE':<40} {'HAM-logP':>10} {'SPAM-logP':>10}  {'PREDICTED':>10}")
    print(SEP2)
    ham_idx  = list(str(c) for c in classes).index("HAM")
    spam_idx = list(str(c) for c in classes).index("SPAM")
    for sentence, pred, lp, pr in zip(sk_test_sentences, preds, log_probs, proba):
        print(f"  {sentence:<40} {lp[ham_idx]:>10.6f} {lp[spam_idx]:>10.6f}  {str(pred):>10}")
    print(SEP2)

    print()
    print()
    for sentence, pred, lp, pr in zip(sk_test_sentences, preds, log_probs, proba):
        print(f"  Sentence        : \"{sentence}\"")
        print(f"  Tokens seen     : {vectorizer.build_analyzer()(sentence)}")
        print(f"  log P(HAM  | doc) = {lp[ham_idx]:.6f}   →  P(HAM  | doc) = {pr[ham_idx]:.6f}")
        print(f"  log P(SPAM | doc) = {lp[spam_idx]:.6f}   →  P(SPAM | doc) = {pr[spam_idx]:.6f}")
        print(f"  Predicted Class : {str(pred)}")
        print(SEP2)
