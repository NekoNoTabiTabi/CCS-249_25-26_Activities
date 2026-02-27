"""
RegEx on NLP - Preprocessing Techniques
Unit 2 Exercise
"""

import re


# =============================================================================
# TASK A: Extract words starting with uppercase letter (10 points)
# =============================================================================


def extract_uppercase_words(text):
    """
    Extract all words starting with an uppercase letter from the text.

    RegEx pattern: r'\b[A-Z][a-zA-Z]*\b'
    - \b       : Word boundary
    - [A-Z]    : First character must be uppercase
    - [a-zA-Z]*: Followed by zero or more letters
    - \b       : Word boundary
    """
    pattern = r"\b[A-Z][a-zA-Z]*\b"
    return re.findall(pattern, text)


def task_a():
    """Task A: Extract uppercase words from Alice in Wonderland text."""
    print("=" * 60)
    print("TASK A: Extract words starting with uppercase")
    print("=" * 60)

    text = """Alice was beginning to get very tired of sitting by her sister on the bank,
and of having nothing to do. Once or twice she had peeped into the book
her sister was reading, but it had no pictures or conversations in it, "and
what is the use of a book," thought Alice, "without pictures or
conversations?"""

    print("\nOriginal Text:")
    print(text)

    pattern = r"\b[A-Z][a-zA-Z]*\b"
    print(f"\nRegEx Pattern: {pattern}")

    uppercase_words = extract_uppercase_words(text)

    print(f"\nWords starting with uppercase ({len(uppercase_words)} found):")
    print(uppercase_words)


# =============================================================================
# TASK B: Extract and replace "whale" variations (15 points)
# =============================================================================


def extract_whale_instances(text):
    """
    Extract all instances of Whale, Whales, whale, whales.

    RegEx pattern: r'\b[Ww]hales?\b'
    - \b       : Word boundary
    - [Ww]     : W or w (case variations)
    - hale     : Literal "hale"
    - s?       : Optional 's' for plural
    - \b       : Word boundary
    """
    pattern = r"\b[Ww]hales?\b"
    return re.findall(pattern, text)


def replace_first_n_whales(text, n=10):
    """
    Replace the first n instances of whale/Whale/whales/Whales with 'leviathan'.

    Uses a counter to track replacements.
    """
    pattern = r"\b[Ww]hales?\b"
    count = [0]  # Using list to allow modification in nested function

    def replacer(match):
        if count[0] < n:
            count[0] += 1
            return "leviathan"
        return match.group(0)

    return re.sub(pattern, replacer, text), count[0]


def task_b():
    """Task B: Extract and replace whale instances from Moby Dick."""
    print("\n" + "=" * 60)
    print("TASK B: Extract and replace 'whale' variations")
    print("=" * 60)

    filename = "melville-moby_dick.txt"

    try:
        with open(filename, "r", encoding="utf-8") as file:
            text = file.read()
        print(f"\nSuccessfully loaded '{filename}'")
    except FileNotFoundError:
        # Try loading from NLTK corpus as fallback
        try:
            import nltk

            nltk.download("gutenberg", quiet=True)
            from nltk.corpus import gutenberg

            text = gutenberg.raw("melville-moby_dick.txt")
            print(f"\nLoaded '{filename}' from NLTK Gutenberg corpus")
        except Exception as e:
            print(f"\nError: Could not load '{filename}'")
            print(f"Please ensure the file exists or install NLTK: {e}")
            return

    pattern = r"\b[Ww]hales?\b"
    print(f"\nRegEx Pattern: {pattern}")

    # Extract all whale instances
    whale_instances = extract_whale_instances(text)
    print(f"\nTotal 'whale' variations found: {len(whale_instances)}")
    print(f"Sample matches (first 20): {whale_instances[:20]}")

    # Replace first 10 instances
    modified_text, replaced_count = replace_first_n_whales(text, n=10)
    print(f"\nReplaced first {replaced_count} instances with 'leviathan'")

    # Show a snippet with replacements
    leviathan_pattern = r".{0,30}leviathan.{0,30}"
    snippets = re.findall(leviathan_pattern, modified_text, re.IGNORECASE)[:5]
    print("\nSample snippets showing replacements:")
    for i, snippet in enumerate(snippets, 1):
        print(f"  {i}. ...{snippet.strip()}...")


# =============================================================================
# TASK C: Extract Jack Sparrow's lines from pirates.txt (15 points)
# =============================================================================


def extract_character_lines(text, character_name):
    """
    Extract all lines spoken by a specific character.

    RegEx pattern for Jack Sparrow: r'JACK SPARROW: (.+)'
    - JACK SPARROW: : Character name followed by colon
    - (.+)          : Capture one or more characters (the dialogue)
    """
    pattern = rf"{character_name}: (.+)"
    return re.findall(pattern, text)


def task_c():
    """Task C: Extract Jack Sparrow's lines from pirates.txt."""
    print("\n" + "=" * 60)
    print("TASK C: Extract Jack Sparrow's lines from pirates.txt")
    print("=" * 60)

    try:
        import nltk

        nltk.download("webtext", quiet=True)
        from nltk.corpus import webtext

        text = webtext.raw("pirates.txt")
        print("\nSuccessfully loaded 'pirates.txt' from NLTK webtext corpus")
    except Exception as e:
        print(f"\nError loading NLTK webtext corpus: {e}")
        print("Please install NLTK and download webtext:")
        print("  pip install nltk")
        print("  python -c \"import nltk; nltk.download('webtext')\"")
        return

    pattern = r"JACK SPARROW: (.+)"
    print(f"\nRegEx Pattern: {pattern}")

    # Extract Jack Sparrow's lines
    jack_lines = extract_character_lines(text, "JACK SPARROW")

    print(f"\nTotal lines spoken by Jack Sparrow: {len(jack_lines)}")
    print("\nJack Sparrow's lines:")
    print("-" * 40)
    for i, line in enumerate(jack_lines, 1):
        print(f"{i:3}. {line}")


# =============================================================================
# MAIN
# =============================================================================


def main():
    """Run all tasks."""
    print("\n" + "#" * 60)
    print("  RegEx on NLP - Preprocessing Techniques")
    print("#" * 60)

    task_a()
    task_b()
    task_c()

    print("\n" + "=" * 60)
    print("All tasks completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
