import re

with open('pirates.txt', 'r', encoding='utf-8', errors='ignore') as file:
    text = file.read()


jack_lines = re.findall(r'^JACK SPARROW:\s*(.+)$', text, re.MULTILINE)

print(f"Total lines spoken by Jack Sparrow: {len(jack_lines)}\n")
print("="*60)
for i, dialogue in enumerate(jack_lines, 1):
    print(f"{i}. {dialogue}")



