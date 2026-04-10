import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    return wordnet.NOUN


def preprocess_text(text):
    text = re.sub(r"\[\d+\]", "", text)

    sentences = sent_tokenize(text)
    processed = []

    for sent in sentences:
        sent = sent.lower()
        sent = re.sub(r"[^a-z0-9\-\s]", " ", sent)

        tokens = word_tokenize(sent)
        pos_tags = pos_tag(tokens)

        cleaned = [
            lemmatizer.lemmatize(tok, get_wordnet_pos(pos))
            for tok, pos in pos_tags
            if len(tok) >= 2
            and not tok.startswith("-")
        ]

        if len(cleaned) >= 3:
            processed.append(cleaned)

    return processed