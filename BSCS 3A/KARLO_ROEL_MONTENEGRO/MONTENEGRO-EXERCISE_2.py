# ELIZA implementation in Python
# Example Generated via ChatGPT
import re
import random
import os

# Global variable to track previous questions (for bonus feature)
previous_questions = []

def reflect(fragment):
    """Reflects user input to make responses more natural."""
    reflections = {
        "am": "are",
        "was": "were",
        "i": "you",
        "i'd": "you would",
        "i've": "you have",
        "i'll": "you will",
        "my": "your",
        "are": "am",
        "you've": "I have",
        "you'll": "I will",
        "your": "my",
        "yours": "mine",
        "you": "me",
        "me": "you",
        "can't": "cannot",
        "can not": "cannot"
    }
    words = fragment.lower().split()
    return ' '.join([reflections.get(word, word) for word in words])

def check_repeated_question(user_input):
    """Checks if the user is repeating a question and returns a sarcastic response."""
    global previous_questions
    
    # Normalize the input for comparison
    normalized_input = user_input.lower().strip()
    
    # Check if this question was asked before
    if normalized_input in previous_questions:
        sarcastic_responses = [
            "Have you tried turning your memory off and on again?",
            "I'm going to assume your Ctrl+C and Ctrl+V keys are stuck.",
            "Oh great, déjà vu! I could have sworn we already covered this...",
            "Are you testing my memory or yours? Because we've been through this already.",
            "I feel like we're in a bad relationship. You never listen to me!",
        ]
        return random.choice(sarcastic_responses)
    
    # Add to history if not repeated
    previous_questions.append(normalized_input)
    return None

def eliza_response(user_input):
    """Generates ELIZA-style responses based on input."""
    
    # BONUS: Check for repeated questions first
    repeated_response = check_repeated_question(user_input)
    if repeated_response:
        return repeated_response
    
    patterns = [
        # Original patterns
        (r"I need (.*)", "Why do you need {0}?"),
        (r"Why don't you (.*)", "Do you really think I don't {0}?"),
        (r"I feel (.*)", "Tell me more about feeling {0}."),
        (r".*want to know.*reasons.*why.*feeling depressed.*", "Feeling depressed all the time is serious. Have you considered exploring what triggers these feelings? When did you first notice this pattern?"),
        (r".*(?:I am|I'm)\s+feeling\s+stressed.*", "Stress can be overwhelming. What do you think is the main source of your stress right now?"),
        (r".*feelings?\s+towards?\s+.*crush.*invalidated.*", "It must be painful to feel that your feelings aren't being acknowledged. Can you tell me more about what happened with your crush?"),
        (r".*[Yy]ou\s+(?:don't|do not)\s+understand\s+me.*", "I'm trying my best to understand you. Can you help me by explaining what I'm missing?"),
        (r".*I\s+(?:can't|cannot)\s+focus\s+on\s+my\s+studies.*", "Difficulty focusing on studies is a common challenge. What distractions or thoughts are preventing you from concentrating?")
    ]
    
    for pattern, response in patterns:
        match = re.match(pattern, user_input, re.IGNORECASE)
        if match:
            # Check if there's a capturing group
            if match.lastindex:
                return response.format(reflect(match.group(1)))
            else:
                return response
    
    return "Can you tell me more?"

# ============================================================================
# TASK 2: RegEx NLP Preprocessing Techniques
# ============================================================================

def task_2a_extract_uppercase_words():
    """
    Task 2a: Extract all words starting with an uppercase letter from the given text.
    RegEx pattern: r'\b[A-Z][a-z]*\b'
    """
    text = """Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do.  Once or twice she had peeped into the book her sister was reading, but it had no pictures or conversations in it, "and what is the use of a book," thought Alice, "without pictures or conversations?"""
    
    # RegEx pattern to match words starting with uppercase letter
    pattern = r'\b[A-Z][a-z]*\b'
    
    # Find all matches
    uppercase_words = re.findall(pattern, text)
    
    print("\n" + "="*80)
    print("TASK 2A: Extract Words Starting with Uppercase Letter")
    print("="*80)
    print(f"RegEx Pattern: {pattern}")
    print(f"\nOriginal Text:\n{text}")
    print(f"\nWords starting with uppercase letter:")
    print(uppercase_words)
    print(f"\nTotal count: {len(uppercase_words)}")
    print("="*80 + "\n")
    
    return uppercase_words


def task_2b_whale_replacement():
    """
    Task 2b: Read melville-moby_dick.txt and extract all instances of 
    Whale/Whales/whale/whales, then replace the first 10 with "leviathan".
    RegEx pattern: r'\b[Ww]hales?\b'
    """
    # Get the file path (same directory as this script)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "melville-moby_dick.txt")
    
    print("\n" + "="*80)
    print("TASK 2B: Extract and Replace Whale/Whales Instances")
    print("="*80)
    
    try:
        # Read the file
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        
        # RegEx pattern to match Whale, Whales, whale, whales
        pattern = r'\b[Ww]hales?\b'
        
        # Find all matches
        all_matches = re.findall(pattern, text)
        
        print(f"RegEx Pattern: {pattern}")
        print(f"\nTotal instances of Whale/Whales/whale/whales found: {len(all_matches)}")
        
        # Count each variation
        from collections import Counter
        match_counts = Counter(all_matches)
        print(f"\nBreakdown:")
        for word, count in sorted(match_counts.items()):
            print(f"  {word}: {count}")
        
        # Show first 10 matches with context
        print(f"\nFirst 10 instances (with context):")
        matches_with_context = list(re.finditer(pattern, text))
        for i, match in enumerate(matches_with_context[:10], 1):
            start = max(0, match.start() - 30)
            end = min(len(text), match.end() + 30)
            context = text[start:end].replace('\n', ' ')
            print(f"  {i}. ...{context}...")
        
        # Replace first 10 instances with "leviathan"
        count = [0]  # Using list to allow modification in nested function
        
        def replace_func(match):
            if count[0] < 10:
                count[0] += 1
                return "leviathan"
            return match.group(0)
        
        modified_text = re.sub(pattern, replace_func, text)
        
        # Save the modified text
        output_path = os.path.join(script_dir, "melville-moby_dick_modified.txt")
        with open(output_path, 'w', encoding='utf-8') as file:
            file.write(modified_text)
        
        print(f"\n✓ Successfully replaced first 10 instances with 'leviathan'")
        print(f"✓ Modified text saved to: melville-moby_dick_modified.txt")
        
        # Verify the replacement
        print(f"\nVerification - First 10 instances after replacement:")
        new_matches = list(re.finditer(r'\bleviathan\b', modified_text))
        for i, match in enumerate(new_matches[:10], 1):
            start = max(0, match.start() - 30)
            end = min(len(text), match.end() + 30)
            context = modified_text[start:end].replace('\n', ' ')
            print(f"  {i}. ...{context}...")
        
        print("="*80 + "\n")
        return all_matches, modified_text
        
    except FileNotFoundError:
        print(f"ERROR: File not found: {file_path}")
        print("Please ensure melville-moby_dick.txt is in the same directory as this script.")
        print("="*80 + "\n")
        return None, None


def task_2c_jack_sparrow_lines():
    """
    Task 2c: Download NLTK, import webtext, load pirates.txt and 
    extract all lines spoken by Jack Sparrow.
    RegEx pattern: r'^.*Jack Sparrow:(.*)$' (multiline mode)
    """
    print("\n" + "="*80)
    print("TASK 2C: Extract Jack Sparrow's Lines from Pirates.txt")
    print("="*80)
    
    try:
        import nltk
        print("✓ NLTK already installed")
    except ImportError:
        print("Installing NLTK package...")
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "nltk"])
        import nltk
        print("✓ NLTK installed successfully")
    
    try:
        # Download webtext corpus if not already downloaded
        print("\nDownloading webtext corpus...")
        nltk.download('webtext', quiet=True)
        
        from nltk.corpus import webtext
        
        # Load pirates.txt
        pirates_text = webtext.raw('pirates.txt')
        
        # RegEx pattern to match lines spoken by Jack Sparrow
        # Pattern variations to catch different formats
        patterns = [
            r'Jack Sparrow:\s*(.+?)(?=\n[A-Z]|\n\n|$)',  # Format: "Jack Sparrow: dialogue"
            r'JACK SPARROW:\s*(.+?)(?=\n[A-Z]|\n\n|$)',  # Format: "JACK SPARROW: dialogue"
            r'JACK:\s*(.+?)(?=\n[A-Z]|\n\n|$)',          # Format: "JACK: dialogue"
        ]
        
        print(f"\nRegEx Patterns used:")
        for i, p in enumerate(patterns, 1):
            print(f"  {i}. {p}")
        
        all_lines = []
        
        # Try each pattern
        for pattern in patterns:
            matches = re.findall(pattern, pirates_text, re.MULTILINE | re.DOTALL)
            all_lines.extend(matches)
        
        # Remove duplicates while preserving order
        unique_lines = []
        seen = set()
        for line in all_lines:
            line_clean = line.strip()
            if line_clean and line_clean not in seen:
                seen.add(line_clean)
                unique_lines.append(line_clean)
        
        print(f"\nTotal lines spoken by Jack Sparrow: {len(unique_lines)}")
        
        if unique_lines:
            print(f"\nFirst 10 lines spoken by Jack Sparrow:")
            for i, line in enumerate(unique_lines[:10], 1):
                # Truncate long lines for display
                display_line = line[:100] + "..." if len(line) > 100 else line
                print(f"  {i}. {display_line}")
            
            if len(unique_lines) > 10:
                print(f"\n  ... and {len(unique_lines) - 10} more lines")
        else:
            print("\nNote: The pirates.txt file may not contain character dialogue in the expected format.")
            print("Showing sample of the text to analyze the format:")
            print("-" * 80)
            print(pirates_text[:1000])
            print("-" * 80)
        
        # Save Jack Sparrow's lines to a file
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(script_dir, "jack_sparrow_lines.txt")
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, line in enumerate(unique_lines, 1):
                f.write(f"{i}. {line}\n\n")
        
        print(f"\n✓ Jack Sparrow's lines saved to: jack_sparrow_lines.txt")
        print("="*80 + "\n")
        
        return unique_lines
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        print("="*80 + "\n")
        return None


def run_eliza():
    """Run the ELIZA chatbot"""
    print("\n" + "="*80)
    print("ELIZA CHATBOT - Interactive Session")
    print("="*80)
    print("ELIZA: Hello! How can I help you today?")
    print("ELIZA: (Type 'quit' or 'exit' to end the conversation)")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["quit", "exit"]:
            print("ELIZA: Goodbye!")
            break
        print(f"ELIZA: {eliza_response(user_input)}")


def main_menu():
    """Main menu to select which task to run"""
    while True:
        print("\nSelect a task to run:")
        print("  1. Run ELIZA Chatbot (Task 1 with 5 new patterns + bonus)")
        print("  2. Task 2a: Extract Uppercase Words")
        print("  3. Task 2b: Whale/Whales Extraction and Replacement")
        print("  4. Task 2c: Extract Jack Sparrow Lines")
        print("  5. Run ALL Tasks (2a, 2b, 2c)")
        print("  6. Exit")
        print("="*80)
        
        choice = input("\nEnter your choice (1-6): ").strip()
        
        if choice == '1':
            run_eliza()
        elif choice == '2':
            task_2a_extract_uppercase_words()
        elif choice == '3':
            task_2b_whale_replacement()
        elif choice == '4':
            task_2c_jack_sparrow_lines()
        elif choice == '5':
            print("\nRunning all RegEx NLP tasks...\n")
            task_2a_extract_uppercase_words()
            task_2b_whale_replacement()
            task_2c_jack_sparrow_lines()
            print("\n✓ All tasks completed!")
        elif choice == '6':
            print("\nExiting program. Goodbye!")
            break
        else:
            print("\nInvalid choice. Please enter a number between 1 and 6.")


# Run the program
if __name__ == "__main__":
    main_menu()
