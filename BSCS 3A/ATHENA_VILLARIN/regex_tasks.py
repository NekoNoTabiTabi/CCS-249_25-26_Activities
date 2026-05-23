# 2. Implementing RegEx on NLP

import re

# Part a: Extract all words starting with an uppercase letter from the text
text = """Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do. Once or twice she had peeped into the book her sister was reading, but it had no pictures or conversations in it, "and what is the use of a book," thought Alice, " without pictures or conversations?"""

pattern_a = r'\b[A-Z][a-z]*\b'
matches_a = re.findall(pattern_a, text)
print("RegEx pattern for a:", pattern_a)
print("Words starting with uppercase:", matches_a)
print()

# Part b: Read melville-moby_dick.txt, extract whale instances, replace first 10 with leviathan
with open('melville-moby_dick.txt', 'r', encoding='utf-8') as f:
    content = f.read()

pattern_b = r'\bwhale[s]?\b'
matches_b = re.findall(pattern_b, content, re.IGNORECASE)
print("RegEx pattern for b:", pattern_b)
print("Number of whale instances found:", len(matches_b))

# Function to replace first n occurrences
def replace_first_n(text, pattern, replacement, n, flags=0):
    count = 0
    def repl(match):
        nonlocal count
        if count < n:
            count += 1
            return replacement
        return match.group(0)
    return re.sub(pattern, repl, text, flags=flags)

modified_content = replace_first_n(content, pattern_b, 'leviathan', 10, re.IGNORECASE)
print("First 10 instances replaced with 'leviathan'.")
print()

# Part c: Load pirates.txt, extract Jack Sparrow's lines
import nltk
nltk.download('webtext', quiet=True)
from nltk.corpus import webtext
pirates = webtext.raw('pirates.txt')
pattern_c = r'^JACK SPARROW:\s*(.*)$'
matches_c = re.findall(pattern_c, pirates, re.MULTILINE | re.IGNORECASE)
print("RegEx pattern for c:", pattern_c)
print("Lines spoken by Jack Sparrow:")
for line in matches_c:
    print(line)