import re

text = """Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do.  Once or twice she had peeped into the book her sister was reading, 
but it had no pictures or conversations in it, "and what is the use of a book," thought Alice, "without pictures or conversations?"""

# Find all words that start with an uppercase letter
pattern1 = r"(?<!\w)[A-Z][a-z]*(?!\w)"
upper_case_words = re.findall(pattern1, text)
print(upper_case_words)
