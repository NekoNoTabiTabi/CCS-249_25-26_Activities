"""
Task 1 (10 points): Dataset Selection and Collection
Select a Wikipedia article and prepare the corpus for training.

The article selected: https://en.wikipedia.org/wiki/JoJo%27s_Bizarre_Adventure
This article should be reasonably long (at least a few thousand words) for good results.
"""

import re
import requests
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize, word_tokenize
from typing import List, Dict
import nltk

# TASK 1: Wikipedia article URL
WIKI_URL = "https://en.wikipedia.org/wiki/JoJo%27s_Bizarre_Adventure"


def ensure_nltk():
    """Download required NLTK data."""
    resources = ["punkt", "punkt_tab"]
    for r in resources:
        try:
            nltk.data.find(f"tokenizers/{r}")
        except LookupError:
            nltk.download(r)


def fetch_wikipedia_article(url: str) -> str:
    """
    TASK 1a: Fetch Wikipedia article content from the specified URL.
    
    Args:
        url: The Wikipedia article URL
        
    Returns:
        Raw text content from the article
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; SkipGram-JOJO-Training/1.0)"
    }
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()

    # Extract main content text from the Wikipedia page
    soup = BeautifulSoup(resp.text, "html.parser")

    content_div = soup.find("div", {"id": "mw-content-text"})
    if content_div is None:
        raise ValueError("Could not find Wikipedia article content.")

    paragraphs = content_div.find_all(["p", "li"])
    text_blocks = []

    for p in paragraphs:
        txt = p.get_text(" ", strip=True)
        if txt:
            text_blocks.append(txt)

    text = "\n".join(text_blocks)

    # Remove citation markers like [1], [2], etc.
    text = re.sub(r"\[[0-9]+\]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_text(text: str) -> List[List[str]]:
    """
    TASK 1b: Preprocess the raw text into sentences and tokens.
    
    Performs:
    - Sentence tokenization
    - Lowercasing
    - Removal of special characters
    - Word tokenization
    - Cleaning short tokens and numbers
    
    Args:
        text: Raw text from the Wikipedia article
        
    Returns:
        List of sentences, where each sentence is a list of cleaned tokens
    """
    sentences = sent_tokenize(text)

    processed = []
    for sent in sentences:
        sent = sent.lower()
        sent = re.sub(r"[^a-z0-9\-\s]", " ", sent)
        sent = re.sub(r"\s+", " ", sent).strip()
        if not sent:
            continue

        tokens = word_tokenize(sent)

        cleaned = []
        for tok in tokens:
            tok = tok.strip("-")
            if not tok:
                continue
            if tok.isdigit():
                continue
            if len(tok) < 2:
                continue
            cleaned.append(tok)

        if len(cleaned) >= 3:
            processed.append(cleaned)

    return processed


def corpus_stats(sentences: List[List[str]]) -> Dict[str, int]:
    """
    TASK 1c: Compute basic statistics about the corpus.
    
    Args:
        sentences: List of tokenized sentences
        
    Returns:
        Dictionary with corpus statistics
    """
    flat = [w for s in sentences for w in s]
    vocab = set(flat)
    return {
        "num_sentences": len(sentences),
        "num_tokens": len(flat),
        "vocab_size": len(vocab),
    }


if __name__ == "__main__":
    print("=== TASK 1: Dataset Selection and Preparation ===\n")
    print(f"Wikipedia URL: {WIKI_URL}\n")
    
    ensure_nltk()
    
    print("Downloading Wikipedia article...")
    raw_text = fetch_wikipedia_article(WIKI_URL)
    print(f"Downloaded {len(raw_text)} characters\n")
    
    print("Preprocessing text...")
    sentences = preprocess_text(raw_text)
    stats = corpus_stats(sentences)
    
    print("=== Corpus Statistics ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    
    # Save preprocessed corpus for use in training
    import pickle
    with open("corpus_sentences.pkl", "wb") as f:
        pickle.dump(sentences, f)
    print("\nSaved corpus to: corpus_sentences.pkl")
