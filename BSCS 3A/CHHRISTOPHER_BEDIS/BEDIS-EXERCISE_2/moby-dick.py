import re

with open("melville-moby_dick.txt", "r", encoding="utf-8") as file:
    text = file.read()

pattern = r"\bwhales?\b"

matches = re.findall(pattern, text, re.IGNORECASE)
print(f"Total matches found: {len(matches)}")

modified_text = re.sub(pattern, "leviathan", text, count=10, flags=re.IGNORECASE)

with open("melville-moby_dick_modified.txt", "w", encoding="utf-8") as file:
    file.write(modified_text)

print("First 10 instances replaced with 'leviathan'.")
