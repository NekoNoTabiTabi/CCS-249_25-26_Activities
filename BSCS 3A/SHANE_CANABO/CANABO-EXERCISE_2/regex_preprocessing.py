# RegEx NLP Preprocessing
# Extract words starting with uppercase letters

import re

# Sample text
text = """Alice was beginning to get very tired of sitting by her sister on the bank,
and of having nothing to do. Once or twice she had peeped into the book
her sister was reading, but it had no pictures or conversations in it, "and
what is the use of a book," thought Alice, "without pictures or
conversations?"""

# RegEx pattern to extract words starting with uppercase letter
pattern = r'\b[A-Z][a-z]*\b'

# Find all matches
uppercase_words = re.findall(pattern, text)

# Display results
print("Words starting with uppercase letter:")
print(uppercase_words)
print(f"\nTotal count: {len(uppercase_words)}")
