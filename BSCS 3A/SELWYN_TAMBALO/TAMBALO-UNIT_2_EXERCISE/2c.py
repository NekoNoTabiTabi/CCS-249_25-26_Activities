import nltk
import re
from nltk.corpus import webtext

nltk.download('webtext')

# Read the text file
text = webtext.raw('pirates.txt')

# Find all lines spoken by Captain Jack Sparrow
jack_lines = []
pattern = re.compile(r'^Jack Sparrow: ', re.IGNORECASE)

for line in text.split('\n'):
    if pattern.match(line):
        line_content = line.split(":", 1)[1].strip()
        jack_lines.append(line_content)

# Print the lines spoken by Captain Jack Sparrow
print(f"Found lines of Captain Jack Sparrow = {len(jack_lines)}\n")
for i, line in enumerate(jack_lines, 1):
    print(f"{i}: {line}")