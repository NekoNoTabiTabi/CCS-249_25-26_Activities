import re

text = "Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do. Once or twice she had peeped into the book her sister was reading, but it had no pictures or conversations in it, &quot;and what is the use of a book,&quot; thought Alice, &quot;without pictures or conversations?"

uppercase = re.findall(r"[A-Z][a-z]+", text)
print("Uppercase letters:", uppercase)
