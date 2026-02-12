# 2. Updating ELIZA. Implementing RegEx on NLP.

import nltk 
import re
nltk.download('webtext')
from nltk.corpus import webtext


# a. (10 points) Extract all of the words starting with an upper case letter from the text:

text = """ Alice was beginning to get very tired of sitting by her sister on the bank,
 and of having nothing to do.  Once or twice she had peeped into the book her sister was reading,
 but it had no pictures or conversations in it, "and what is the use of a book," 
 thought Alice, "without pictures or conversations?
"""

uppercase_words = re.findall(r"[A-Z][a-z]*", text)
print(uppercase_words)

#------------------------------------------------------------------------#

# b. (10 points) Read the “melville-moby_dick.txt” into a Python program and extract all of the instances of the word whale in said source.

with open(
    r"C:\Users\jessi\PORRAS_NLP_Repo\CCS-249-Sample-Codes\Unit 2\melville-moby_dick.txt",
    "r",
    encoding="utf-8"
) as file:
    text = file.read()

whale_instances = re.findall(r"\bwhales?\b", text, re.IGNORECASE)
count = len(whale_instances)

print(whale_instances)
print(count)

#------------------------------------------------------------------------#

# c. (10 points) Download the NLTK package into your local environment and import webtext (from nltk.corpus import webtext).  
# Load the pirates.txt and  extract all of the lines spoken by the character Jack Sparrow.
 
pirate_text = webtext.raw('pirates.txt')
print(pirate_text[:500])

jack_sparrow = re.findall(r'^JACK SPARROW:\s*(.*)$', pirate_text, re.MULTILINE)

sparrow_count = len(jack_sparrow)

for line in jack_sparrow:
    print(line)

print(sparrow_count)

