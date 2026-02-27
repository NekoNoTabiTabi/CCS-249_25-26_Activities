import re

#Read the file
with open("melville-moby_dick.txt", "r", encoding="utf-8") as file:
    text = file.read()

#Extract all occurrences of whale / whales (case-insensitive)
pattern = r"\bwhales?\b"
matches = re.findall(pattern, text, flags=re.IGNORECASE)

print("Total whale-related words found:", len(matches))

#Replace only the first 10 occurrences with 'leviathan'
modified_text = re.sub(
    pattern,
    "leviathan",
    text,
    count=10,
    flags=re.IGNORECASE
)

#Save the modified text to a new file
with open("melville-moby_dick_modified.txt", "w", encoding="utf-8") as file:
    file.write(modified_text)

print("First 10 occurrences replaced with 'leviathan'.")
