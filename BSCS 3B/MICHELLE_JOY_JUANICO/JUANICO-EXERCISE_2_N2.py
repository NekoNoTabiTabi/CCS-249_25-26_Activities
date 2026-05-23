import re
import os

text = "Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do. Once or twice she had peeped into the book her sister was reading, but it had no pictures or conversations in it, \"and what is the use of a book,\" thought Alice, \"without pictures or conversations?"

# a. Extract all of the words starting with an upper case letter from the text
capitalized_words = re.findall(r'\b[A-Z][a-zA-Z]*\b', text)
print("Capitalized Words:", capitalized_words)

# b. Read the “melville-moby_dick.txt” into a Python program and extract all of the instances of the word whale in said source.
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'melville-moby_dick.txt')
with open(file_path, "r", encoding='UTF-8') as file:
    moby_dick_text = file.read()
whales = re.findall(r'\bwhales?\b', moby_dick_text, re.IGNORECASE)
print("Number of whales:", len(whales))
print("Whales:", whales)

# c. Download the NLTK package into your local environment and import webtext (from nltk.corpus import webtext). Load the pirates.txt and extract all of the lines spoken by the character Jack Sparrow.
from nltk.corpus import webtext
pirates_lines = webtext.raw('pirates.txt').split('\n')
jack_sparrow_lines = re.findall(r'Jack Sparrow:.*', webtext.raw('pirates.txt'), re.IGNORECASE)
print("Jack Sparrow Lines:", jack_sparrow_lines)

# save to text file with one dialogue line per line
with open("jack_sparrow_lines.txt", "w", encoding="utf-8") as file:
    for line in jack_sparrow_lines:
        file.write(line + "\n")
