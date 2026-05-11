import re
import nltk

text = """Alice was beginning to get very tired of sitting by her sister on the bank,
and of having nothing to do. Once or twice she had peeped into the book
her sister was reading, but it had no pictures or conversations in it, "and
what is the use of a book," thought Alice, "without pictures or
conversations?"""

a = re.findall(r"\b[A-Z][a-z]*\b", text)
print(a)


with open("C:\\College\\3rd-Year\\2nd\\codes\\nlp\\CCS-249-Sample-Codes\\Unit 2\\melville-moby_dick.txt", "r", encoding="utf-8") as f:
    text2 = f.read()

def whale_sub(match):
    word = match.group()
    
    if word[0].isupper():
        replacement = "Leviathan"
    else:
        replacement = "leviathan"
    if word.lower().endswith("s"):
        replacement += "s"
    return replacement

b = re.sub(r"\b[Ww]hale\b|\b[Ww]hales\b", whale_sub, text2)





nltk.download('webtext')
from nltk.corpus import webtext
pirates = webtext.raw('pirates.txt')  # entire text as a string

lines = pirates.split('\n')
jack_lines = [line for line in lines if line.startswith("JACK SPARROW:")]
print("Jack Sparrow's lines:\n")

print("pirates_local.txt has been saved!")

for line in jack_lines:
    print(line)


