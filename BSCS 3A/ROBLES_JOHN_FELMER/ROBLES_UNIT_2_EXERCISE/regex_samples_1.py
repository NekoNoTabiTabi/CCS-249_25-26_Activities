# SIMPLE SEARCH

import re

text = """
Alice was beginning to get very tired of sitting by her sister on the bank,
and of having nothing to do. Once or twice she had peeped into the book
her sister was reading, but it had no pictures or conversations in it, "and
what is the use of a book," thought Alice, "without pictures or conversations?"
"""

# Extract words that start with an uppercase letter
pattern = r"\b[A-Z][a-z]*\b"

uppercase_words = re.findall(pattern, text)

print("Words starting with an uppercase letter:")
print(uppercase_words)


# Simple substring pattern
match = re.search(pattern, text)

if match:
    print("Found:", match.group(), "at", match.span())
else:
    print("Not found")

# CHARACTER CLASSES AND RANGES []

text = "Mary married mary in 2024, on page 3 of Chapter 1."

# [mM]ary  -> matches 'Mary' or 'mary'
for m in re.finditer(r"[mM]ary", text):
    print("Name match:", m.group())

# [0-9] or \d -> any single digit
digits = re.findall(r"[0-9]", text)
print("Digits (via [0-9]):", digits)

digits_alias = re.findall(r"\d", text)   # convenient alias
print("Digits (via \\d):", digits_alias)

# [A-Z] and [a-z]
upper_letters = re.findall(r"[A-Z]", text)
lower_letters = re.findall(r"[a-z]", text)
print("Uppercase letters:", upper_letters)
print("Lowercase letters (first 10):", lower_letters[:10])


# NEGATION

text = "The End.\nOyfn pripetchik\nI have no exquisite reason."

# [^A-Z] : any character that is NOT an uppercase letter
not_upper = re.findall(r"[^A-Z]", text)
print("Number of non-uppercase characters:", len(not_upper))

# [^Ss] : any character that is not 'S' or 's'
not_s = re.findall(r"[^Ss]", "Ss snakes and ladders")
print("Characters that are not S or s:", not_s[:15])

# [^.] : any character that is not a period
no_dots = re.findall(r"[^.]", "Our resident Djinn.")
print("String without periods:", "".join(no_dots))

# DISJUNCTION

text = "Groundhog is another name for woodchuck! Yours or mine? a, b, c."

# groundhog|woodchuck
animals = re.findall(r"groundhog|woodchuck", text, flags=re.IGNORECASE)
print("Animals:", animals)

# yours|mine
pronouns = re.findall(r"yours|mine", text, flags=re.IGNORECASE)
print("Pronouns:", pronouns)

# a|b|c is equivalent to [abc]
letters1 = re.findall(r"a|b|c", text)
letters2 = re.findall(r"[abc]", text)
print("a|b|c:", letters1)
print("[abc]:", letters2)

# [gG]roundhog|[Ww]oodchuck : case-flexible
case_flexible = re.findall(r"[gG]roundhog|[Ww]oodchuck", text)
print("Case-flexible animals:", case_flexible)


# CONVENTIONAL ALIASES

text = "Fahrenheit 451\nBlue Moon\tDaiyu_123!\nLook up  \n"

digits      = re.findall(r"\d", text)   # any digit
nondigits   = re.findall(r"\D", text)   # any non-digit
wordchars   = re.findall(r"\w", text)   # letters, digits, underscore
nonword     = re.findall(r"\W", text)   # not \w
whitespace  = re.findall(r"\s", text)   # space, tab, newline, etc.
nonspace    = re.findall(r"\S", text)   # not whitespace

print("Digits:", digits)
print("Some non-digits:", nondigits[:15])
print("Word chars:", wordchars[:15])
print("Non-word chars:", nonword[:10])
print("Whitespace chars (repr):", [repr(c) for c in whitespace])
print("Non-whitespace chars:", nonspace[:15])


# SUBSTITUTION

string = "I bought cherry pie. cherry is my favorite fruit."

# Simple substitution: cherry -> apricot
result = re.sub(r"cherry", r"apricot", string)
print(result)

# Capitalize 'janet'
text = "janet likes cherries. janet also likes apples."
result2 = re.sub(r"janet", r"Janet", text)
print(result2)


# CAPTURE GROUPS

import re

text = "The date is 10/15/2011. Another date is 01/02/2020."

# US to EU date: (mm/dd/yyyy) -> (dd-mm-yyyy)
pattern = r"(\d{2})/(\d{2})/(\d{4})"
repl    = r"\2-\1-\3"   # use groups 1, 2, 3 in new order

converted = re.sub(pattern, repl, text)
print("Original:", text)
print("Converted:", converted)

# LOOKAHEAD

lines = [
    "Today is sunny",
    "tomorrow is rainy",
    "Monday begins",
    "tuesday begins"
]

# Capture the first word, only if it does NOT start with T or t
pattern = r"^(?![tT])(\w+)\b"

for line in lines:
    m = re.search(pattern, line)
    if m:
        print("Accepted first word:", m.group(1), "in:", repr(line))
    else:
        print("Rejected line:", repr(line))