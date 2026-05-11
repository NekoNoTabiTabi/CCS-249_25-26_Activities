import re
import nltk
from nltk.corpus import webtext


print("--- TASK A ---")

text_a = """Alice was beginning to get very tired of sitting by her sister on the bank,
and of having nothing to do.  Once or
twice she had peeped into the book her sister was reading, but it had no
pictures or conversations in it, "and what is the use of a book,"
thought Alice, "without pictures or conversations?"""

# Regex Pattern:
pattern_a = r"[A-Z]\w+"
print(f"Regex Pattern: {pattern_a}")

matches_a = re.findall(pattern_a, text_a)
print("Words starting with uppercase:", matches_a)
print("\n")


print("--- TASK B ---")

#Regex Pattern:
pattern_b = r"[Ww]hales?"
print(f"Regex Pattern: {pattern_b}")

try:
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, 'melville-moby_dick.txt')
    with open(file_path, 'r', encoding='utf-8') as f:
        content_b = f.read()

    whales = re.findall(pattern_b, content_b)
    print(f"Found {len(whales)} instances of 'whale' variations.")

    new_content_b = re.sub(pattern_b, "leviathan", content_b, count=10)
    
    print("Replacement complete (first 10 instances).")
    
except FileNotFoundError:
    print("Error: 'melville-moby_dick.txt' not found in the directory.")
    print("Please ensure the file exists to run this part of the code.")

print("\n")


print("--- TASK C ---")

try:
    nltk.data.find('corpora/webtext')
except LookupError:
    nltk.download('webtext')


pirates_text = webtext.raw('pirates.txt')

#Regex Pattern:
# ^             = Start of a line
# JACK SPARROW: = Literal name
# (.*)          = Capture group for the dialogue
pattern_c = r"^JACK SPARROW:(.*)"
print(f"Regex Pattern: {pattern_c}")

jack_lines = re.findall(pattern_c, pirates_text, re.MULTILINE)

print(f"Number of lines spoken by Jack Sparrow: {len(jack_lines)}")
print("First 5 lines:")
for line in jack_lines[:5]:
    print(f" - {line.strip()}")