# RegEx NLP Preprocessing - Extract Jack Sparrow's Lines
# Using NLTK webtext corpus

import re
import nltk

nltk.download('webtext')


from nltk.corpus import webtext

pirates_text = webtext.raw('pirates.txt')

print("=" * 60)
print("EXTRACTING JACK SPARROW'S LINES")
print("=" * 60)

# RegEx pattern to match Jack Sparrow's dialogue
# Pattern matches lines like "JACK:" or "JACK SPARROW:" followed by dialogue
pattern = r'^JACK(?:\s+SPARROW)?:\s*(.+)$'

lines = pirates_text.split('\n')
jack_lines = []

for line in lines:
    match = re.match(pattern, line, re.IGNORECASE)
    if match:
        dialogue = match.group(1).strip()
        jack_lines.append(dialogue)

# Display results
print(f"\nTotal lines spoken by Jack Sparrow: {len(jack_lines)}")
print("\n" + "=" * 60)
print("FIRST 10 LINES:")
print("=" * 60)
for i, line in enumerate(jack_lines[:10], 1):
    print(f"{i}. {line}")

# Save Jack's lines to a file
output_file = 'jack_sparrow_lines.txt'
with open(output_file, 'w', encoding='utf-8') as f:
    f.write("JACK SPARROW'S LINES FROM PIRATES.TXT\n")
    f.write("=" * 60 + "\n\n")
    for i, line in enumerate(jack_lines, 1):
        f.write(f"{i}. {line}\n")

print("\n" + "=" * 60)
print(f"All lines saved to: {output_file}")
print(f"\nRegEx pattern used: {pattern}")
print("=" * 60)
