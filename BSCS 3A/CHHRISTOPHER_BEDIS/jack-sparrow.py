import re
from nltk.corpus import webtext

text = webtext.raw('pirates.txt')

pattern = r'^\s*JACK SPARROW:.*$'

# Extract all lines
jack_lines = re.findall(pattern, text, re.MULTILINE | re.IGNORECASE)

print(f"Total Jack Sparrow lines: {len(jack_lines)}")
