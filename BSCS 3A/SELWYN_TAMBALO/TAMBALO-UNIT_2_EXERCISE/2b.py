import re

# Read the file
file = open("/home/sprakle/Documents/NLP/melville-moby_dick.txt", "r", encoding="utf-8")
text = file.read()
file.close()

# Find all instances of "whale" or "Whale" or "whales" or "Whales"
pattern = r"(?<!\w)[wW]hales?(?!\w)"
all_instances = re.findall(pattern, text)
print(f"Total instances found = {len(all_instances)}")

# Replace first 10 instances of "whale" or "Whale" or "whales" or "Whales"
def replace_whale(match):
    count = 0
    count += 1
    if count <= 10:
        return "leviathan"
    else:
        return match.group(0)

new_text = re.sub(pattern, replace_whale, text)

# Write the modified text to a new file
output = open("/home/sprakle/Documents/NLP/modified_melville-moby_dick.txt", "w", encoding="utf-8")
output.write(new_text)
output.close()

print("First 10 instances replaced successfully!")