import nltk
from nltk.corpus import webtext

# Download required NLTK data (run once)
nltk.download('webtext')

# Load pirates.txt
pirates_text = webtext.raw('pirates.txt')

# Split the text into individual lines
lines = pirates_text.split('\n')

# Extract lines spoken by Jack Sparrow
jack_sparrow_lines = [
    line for line in lines
    if line.startswith("JACK SPARROW")
]

# Print the extracted lines
print("Lines spoken by Jack Sparrow:\n")
for line in jack_sparrow_lines:
    print(line)
