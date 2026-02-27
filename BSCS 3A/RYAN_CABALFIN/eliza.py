# Unit 2 – Exercise 1
# Name: Ryan Cabalfin
# Section: CS 3A
# ELIZA implementation in Python
# Example Generated via ChatGPT
import re

previous_inputs = set()

def normalize(text: str) -> str:
    """
    Normalize text so repeated questions can be detected even if user changes
    capitalization or adds punctuation/spaces.
    """
    text = text.lower().strip()
    # Remove non-word characters (keeps letters/numbers/underscore)
    text = re.sub(r"\W+", "", text)
    return text

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
        "me": "you"
    }
    words = fragment.lower().split()
    return ' '.join([reflections.get(word, word) for word in words])

def eliza_response(user_input):
    """Generates ELIZA-style responses based on input."""

    # --- BONUS: repeated question detection ---
    norm = normalize(user_input)
    if norm in previous_inputs:
        return "You already asked that. Are you trying to confuse me on purpose?"
    previous_inputs.add(norm)

    patterns = [
        # Existing patterns (fixed to support both don't/don’t)
        (r"^I need (.*)$", "Why do you need {0}?"),
        (r"^Why (?:don't|don’t) you (.*)$", "Do you really think I don't {0}?"),
        (r"^I feel (.*)$", "Tell me more about feeling {0}."),

        # --- REQUIRED: 5 new patterns ---

        # a) "I want to know the reasons why I am feeling depressed all the time."
        (r"^I want to know the reasons why I am feeling depressed all the time\.?$",
         "What do you think might be the reasons you feel depressed all the time?"),

        # b) "I am feeling stressed."
        (r"^I am feeling stressed\.?$",
         "What do you think is causing you to feel stressed?"),

        # c) "My feelings towards my crush are invalidated."
        (r"^My feelings towards my crush are invalidated\.?$",
         "How does it feel when your feelings are invalidated?"),

        # d) "You don’t understand me" OR "You do not understand me"
        (r"^You (?:don't|don’t|do not) understand me\.?$",
         "What makes you think I don't understand you?"),

        # e) "I can't focus on my studies" OR "I cannot focus on my studies"
        (r"^I (?:can't|cannot) focus on my studies\.?$",
         "What do you think is making it hard for you to focus on your studies?")
    ]

    for pattern, response in patterns:
        match = re.match(pattern, user_input, re.IGNORECASE)
        if match:
            # Only patterns with (.*) groups will have group(1)
            if match.lastindex:
                return response.format(reflect(match.group(1)))
            return response

    return "Can you tell me more?"

print("ELIZA: Hello! How can I help you today?")
while True:
    user_input = input("You: ")
    if user_input.lower() in ["quit", "exit"]:
        print("ELIZA: Goodbye!")
        break
    print(f"ELIZA: {eliza_response(user_input)}")