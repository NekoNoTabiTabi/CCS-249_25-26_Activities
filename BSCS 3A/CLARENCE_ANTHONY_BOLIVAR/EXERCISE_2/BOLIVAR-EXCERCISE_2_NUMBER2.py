# NLP RegEx Preprocessing Techniques
import re

# Text sample
text = """Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do.  Once or twice she had peeped into the book her sister was reading, but it had no pictures or conversations in it, "and what is the use of a book," thought Alice, "without pictures or conversations?"""

# a. Extract all words starting with an uppercase letter
print("=" * 60)
print("a. Words starting with an uppercase letter:")
print("=" * 60)

# Pattern: \b[A-Z][a-z]* matches words that start with uppercase letter
# \b = word boundary
# [A-Z] = uppercase letter
# [a-z]* = zero or more lowercase letters following
uppercase_words = re.findall(r'\b[A-Z][a-z]*', text)

print(f"Pattern used: r'\\b[A-Z][a-z]*'")
print(f"\nWords found: {uppercase_words}")
print(f"Total count: {len(uppercase_words)}")
print(f"Unique words: {set(uppercase_words)}")

# b. Extract all instances of Whale/Whales/whale/whales and replace first 10 with "leviathan"
print("\n" + "=" * 60)
print("b. Extract and replace whale/whales instances:")
print("=" * 60)

# Read the Moby Dick file
try:
    with open(r'c:\Users\linag\Desktop\melville-moby_dick.txt', 'r', encoding='utf-8') as file:
        moby_dick_text = file.read()
    
    # Pattern to match Whale, Whales, whale, whales (case-sensitive individual matches)
    # Using \b for word boundaries to avoid partial matches
    whale_pattern = r'\b[Ww]hales?\b'
    
    # Find all instances
    whale_matches = re.findall(whale_pattern, moby_dick_text)
    
    print(f"Pattern used: r'\\b[Ww]hales?\\b'")
    print(f"\nTotal instances found: {len(whale_matches)}")
    
    # Count each variation
    from collections import Counter
    whale_counts = Counter(whale_matches)
    print(f"\nBreakdown by variation:")
    for word, count in sorted(whale_counts.items()):
        print(f"  {word}: {count}")
    
    # Show first 20 instances as examples
    print(f"\nFirst 20 instances: {whale_matches[:20]}")
    
    # Replace first 10 instances with "leviathan"
    replacement_count = 0
    def replace_first_10(match):
        global replacement_count
        if replacement_count < 10:
            replacement_count += 1
            return "leviathan"
        return match.group(0)
    
    modified_text = re.sub(whale_pattern, replace_first_10, moby_dick_text)
    
    print(f"\n✓ Replaced first 10 instances with 'leviathan'")
    
    # Save the modified text
    output_file = r'c:\Users\linag\Desktop\melville-moby_dick_modified.txt'
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(modified_text)
    
    print(f"✓ Modified text saved to: melville-moby_dick_modified.txt")
    
    # Verify the replacements by showing context around first few replacements
    print(f"\nVerification - Context around first 3 replacements:")
    leviathan_pattern = r'.{0,50}leviathan.{0,50}'
    leviathan_contexts = re.findall(leviathan_pattern, modified_text, re.IGNORECASE)
    for i, context in enumerate(leviathan_contexts[:3], 1):
        print(f"\n{i}. ...{context.strip()}...")

except FileNotFoundError:
    print("Error: melville-moby_dick.txt file not found!")
except Exception as e:
    print(f"Error: {e}")

# c. Extract all lines spoken by Jack Sparrow from pirates.txt
print("\n" + "=" * 60)
print("c. Extract Jack Sparrow's lines from pirates.txt:")
print("=" * 60)

try:
    import nltk
    
    # Download webtext corpus if not already downloaded
    try:
        nltk.data.find('corpora/webtext')
    except LookupError:
        print("Downloading webtext corpus...")
        nltk.download('webtext', quiet=True)
    
    from nltk.corpus import webtext
    
    # Load pirates.txt
    pirates_text = webtext.raw('pirates.txt')
    
    print(f"Loaded pirates.txt ({len(pirates_text)} characters)")
    
    # Pattern to match lines spoken by Jack Sparrow
    # Looking for variations like:
    # JACK: dialogue
    # JACK SPARROW: dialogue
    # Captain Jack Sparrow: dialogue
    # Typically dialogue scripts use format: CHARACTER: line
    
    # Split text into lines and find Jack's lines
    jack_pattern = r'^(JACK|JACK SPARROW|Captain Jack Sparrow|CAPTAIN JACK SPARROW)\s*:(.+)$'
    
    lines = pirates_text.split('\n')
    jack_lines = []
    
    for line in lines:
        match = re.match(jack_pattern, line.strip(), re.IGNORECASE)
        if match:
            jack_lines.append(match.group(2).strip())
    
    print(f"\nPattern used: r'^(JACK|JACK SPARROW|Captain Jack Sparrow|CAPTAIN JACK SPARROW)\\s*:(.+)$'")
    print(f"Total lines spoken by Jack Sparrow: {len(jack_lines)}")
    
    # Show first 10 lines
    print(f"\nFirst 10 lines by Jack Sparrow:")
    for i, line in enumerate(jack_lines[:10], 1):
        print(f"{i}. {line}")
    
    # Save all Jack's lines to a file
    output_file = r'c:\Users\linag\Desktop\jack_sparrow_lines.txt'
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write("Jack Sparrow's Lines from pirates.txt\n")
        file.write("=" * 60 + "\n\n")
        for i, line in enumerate(jack_lines, 1):
            file.write(f"{i}. {line}\n")
    
    print(f"\n✓ All Jack Sparrow's lines saved to: jack_sparrow_lines.txt")
    
    # Statistics
    if jack_lines:
        total_words = sum(len(line.split()) for line in jack_lines)
        avg_words = total_words / len(jack_lines)
        print(f"\nStatistics:")
        print(f"  Total words spoken: {total_words}")
        print(f"  Average words per line: {avg_words:.1f}")
        print(f"  Longest line: {max(jack_lines, key=len)[:100]}...")
    
except ImportError:
    print("Error: NLTK package not installed. Run: pip install nltk")
except Exception as e:
    print(f"Error: {e}")
