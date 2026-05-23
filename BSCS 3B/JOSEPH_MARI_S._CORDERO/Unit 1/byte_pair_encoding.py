from collections import Counter

def get_stats(vocab):
    """Compute the frequency of adjacent symbol pairs in the vocabulary."""
    pairs = Counter()
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[symbols[i], symbols[i + 1]] += freq
    return pairs

def merge_vocab(pair, vocab):
    """Merge the most frequent pair in the vocabulary."""
    new_vocab = {}
    bigram = ' '.join(pair)
    replacement = ''.join(pair)
    for word in vocab:
        new_word = word.replace(bigram, replacement)
        new_vocab[new_word] = vocab[word]
    return new_vocab

def byte_pair_encoding(corpus, num_merges=10):
    """Performs Byte Pair Encoding (BPE) on the given corpus."""
    vocab = Counter(corpus)
    vocab = {" ".join(word): freq for word, freq in vocab.items()}  # Add spaces to split characters
    
    for _ in range(num_merges):
        pairs = get_stats(vocab)
        if not pairs:
            break
        best_pair = max(pairs, key=pairs.get)
        vocab = merge_vocab(best_pair, vocab)
        print(f"Merging: {best_pair} -> {''.join(best_pair)}")
    
    return vocab

# corpus = ["low", "lower", "newest", "widest"]
input = 'the cat sat on the mat the dog ran in the park cats and dogs are pets the cat chased the dog'
corpus = input.split()
vocab = byte_pair_encoding(corpus, num_merges=10)
print("Final Vocabulary:", vocab)
