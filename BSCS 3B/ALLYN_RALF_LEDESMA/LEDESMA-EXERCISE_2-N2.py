import re


# a. (10 points) Extract all of the words starting with an upper case letter from the text:
text = """
Alice was beginning to get very tired of sitting by her sister on the bank,
and of having nothing to do. Once or twice she had peeped into the book
her sister was reading, but it had no pictures or conversations in it, &quot;and
what is the use of a book, "thought Alice, "without pictures or
conversations?
"""


def extract_uppercase_words(text):
    """Extracts all uppercase words from the given text."""
    pattern = r'\b[A-Z]\w*\b'
    uppercase_words = re.findall(pattern, text)
    return uppercase_words

output = extract_uppercase_words(text)
# print(output)


# b. (10 points) Read the “melville-moby_dick.txt” into a Python program and extract all of the instances of the word whale in said source.
def extract_whale_instances(filename):
    """Extracts all instances of the word 'whale' from a file."""
    with open(filename, 'r', encoding='utf-8') as file:
        content = file.read()

    # Case-insensitive search for the word "whale"
    pattern = r'\bwhales*\b'
    whale_instances = re.findall(pattern, content, re.IGNORECASE)
    return whale_instances

whale_output = extract_whale_instances('melville-moby_dick.txt')
# print(f"Total instances of 'whale': {len(whale_output)}")
# print(f"First 10 instances: {whale_output[:10]}")


# c. (10 points) Download the NLTK package into your local environment
# and import webtext (from nltk.corpus import webtext). Load the
# pirates.txt and extract all of the lines spoken by the character Jack Sparrow.
import nltk

# Uncomment the line below to download webtext corpus (only needed once)
# nltk.download('webtext')

try:
    from nltk.corpus import webtext

    def extract_jack_sparrow_lines():
        """Extracts all lines spoken by Jack Sparrow from pirates.txt."""
        pirates_text = webtext.raw('pirates.txt')
        jack_sparrow_lines = re.findall(r'Jack Sparrow:.*', pirates_text, re.IGNORECASE)
        return jack_sparrow_lines
    
    jack_sparrow_output = extract_jack_sparrow_lines()
    print("Jack Sparrow Lines:", jack_sparrow_output)
    
except Exception as e:
    print(f"Error loading NLTK webtext: {e}")
    print("Make sure to uncomment nltk.download('webtext') and run it once first.")

