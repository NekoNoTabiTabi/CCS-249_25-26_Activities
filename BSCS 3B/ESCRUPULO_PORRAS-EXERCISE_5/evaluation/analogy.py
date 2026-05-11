def evaluate_analogies(model, analogies):
    results = []

    for a, b, c, d in analogies:
        try:
            preds = model.wv.most_similar(
                positive=[b, c],
                negative=[a],
                topn=5
            )
            words = [w for w, _ in preds]
            results.append((a, b, c, d, words))
        except KeyError:
            continue

    return results