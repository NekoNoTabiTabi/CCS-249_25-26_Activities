# ELIZA implementation in Python
import re

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
        "you've": "I have",
        "you'll": "I will",
        "your": "my",
        "yours": "mine",
        "you": "me",
        "me": "you"
    }

    words = fragment.lower().split()
    reflected = []
    for i, word in enumerate(words):
        # handle 'are' -> 'am' only when preceding pronoun is 'you' (or contractions)
        if word == 'are':
            prev = words[i-1] if i > 0 else ''
            if prev in ("you", "you're", "youre", "you've", "youll", "you'll"):
                reflected.append('am')
                continue
        # handle 'am' -> 'are' only when preceding pronoun is 'i' (or contractions)
        if word == 'am':
            prev = words[i-1] if i > 0 else ''
            if prev in ("i", "i'm", "im"):
                reflected.append('are')
                continue

        reflected.append(reflections.get(word, word))

    return ' '.join(reflected)

def eliza_response(user_input):
    """Generates ELIZA-style responses based on input."""
    patterns = [
        (r"I need (.*)", "Why do you need {0}?"),
        (r"Why don’t you (.*)", "Do you really think I don't {0}?"),
        (r"I feel (.*)", "Tell me more about feeling {0}."),
        (r"I want to know (.*)", "What would it mean to you to know {0}?"),
        (r"I am feeling (.*)", "What made you feel {0}?"),
        (r"My feelings towards my crush are (.*)", "Do you often feel {0}?"),
        (r"You (?:don't|do not) understand (.*)", "What makes you think I don't understand {0}?"),
        (r"I (?:can't|cannot) focus (.*)", "What is preventing you from focusing {0}?"),
    ]
    
    for pattern, response in patterns:
        match = re.match(pattern, user_input, re.IGNORECASE)
        # print(match)
        if match:
            return response.format(reflect(match.group(1)))
    
    return "Can you tell me more?"

print("ELIZA: Hello! How can I help you today?")
previous_input = ""
while True:
    user_input = input("You: ")
    
    # if repeating input, do sarcastic response
    if user_input == previous_input:
        print("ELIZA: Are you seriously repeating yourself?")
        continue
    
    if user_input.lower() in ["quit", "exit"]:
        print("ELIZA: Goodbye!")
        break
    
    print(f"ELIZA: {eliza_response(user_input)}")
    previous_input = user_input
