import re

# Sample string
simple_example = "The quick brown fox jumps over the lazy dog near the bank of the river."

# Manual implementation of tokenization
# Space based tokenization
space_tokenization = simple_example.split(" ")
print(space_tokenization)

# Using regex to match all of the text, without the punctuations
words_only = r"\w+"
re_tokenize = re.findall(words_only, simple_example)
print(re_tokenize)

# Using regex to tokenize sentences
sentences = "Hello! I would like to say to you that. You know?"
sentence_re = r"\w+[!.?]|\w+"
sentence_tokenize = re.findall(sentence_re, sentences)
print(sentence_tokenize)

# How can you remove the punctuation from the words?