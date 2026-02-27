# RegEx NLP Preprocessing - Whale Extraction and Replacement
import re

# Read the Moby Dick text file
with open('melville-moby_dick.txt', 'r', encoding='utf-8') as file:
    text = file.read()

# RegEx pattern to match whale/whales (case-insensitive)
pattern = r'\b[Ww]hales?\b'

# Extract all instances of whale/whales
all_whales = re.findall(pattern, text)

print("=" * 60)
print("EXTRACTED WHALE INSTANCES")
print("=" * 60)
print(f"Total instances found: {len(all_whales)}")
print(f"\nFirst 20 instances: {all_whales[:20]}")
print(f"\nBreakdown:")
print(f"  'Whale': {all_whales.count('Whale')}")
print(f"  'Whales': {all_whales.count('Whales')}")
print(f"  'whale': {all_whales.count('whale')}")
print(f"  'whales': {all_whales.count('whales')}")

# Replace the first 10 instances with "leviathan"
replacement_count = 0
def replace_first_10(match):
    global replacement_count
    if replacement_count < 10:
        replacement_count += 1
        return "leviathan"
    return match.group(0)

modified_text = re.sub(pattern, replace_first_10, text)

# Save to a new file
output_file = 'melville-moby_dick_modified.txt'
with open(output_file, 'w', encoding='utf-8') as file:
    file.write(modified_text)

print("\n" + "=" * 60)
print("REPLACEMENT COMPLETE")
print("=" * 60)
print(f"Replaced first 10 instances with 'leviathan'")
print(f"Modified text saved to: {output_file}")
print(f"\nRegEx pattern used: {pattern}")
