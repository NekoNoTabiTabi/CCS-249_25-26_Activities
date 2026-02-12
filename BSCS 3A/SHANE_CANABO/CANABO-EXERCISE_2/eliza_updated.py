# ELIZA implementation in Python
# Example Generated via ChatGPT
import re
import random

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

# Track previous questions
previous_questions = []

def eliza_response(user_input):
    """Generates ELIZA-style responses based on input."""
    
    # Check for repeated questions
    normalized_input = user_input.lower().strip()
    if normalized_input in previous_questions:
        sarcastic_responses = [
            "Really? You already asked me that. Are we going in circles now?",
            "Didn't we just discuss this? Short-term memory issues?",
            "You literally just asked that. Perhaps you need a break?",
            "Again with the same question? I'm beginning to question your creativity.",
            "We've been through this already. Were you even listening to my response?"
        ]
        return random.choice(sarcastic_responses)
    
    # Add to history
    previous_questions.append(normalized_input)
    
    patterns = [
        (r".*want to know.*why.*feeling (.*)", "What do you think contributes to feeling {0}?"),
        (r".*feeling stressed(.*)", "What do you think is making you feel stressed{0}?"),
        (r".*feelings.*crush.*are (.*)", "Why do you feel your feelings toward your crush are {0}?"),
        (r"You (?:don.?t|do not) understand me(.*)", "I want to understand you better{0}. What am I missing?"),
        (r"I can(?:not|.?t) focus on my studies(.*)", "What is making it hard to focus on your studies{0}?"),
        (r"I need (.*)", "Why do you need {0}?"),
        (r"Why don’t you (.*)", "Do you really think I don't {0}?"),
        (r"I feel (.*)", "Tell me more about feeling {0}.")
    ]
    
    for pattern, response in patterns:
        match = re.match(pattern, user_input, re.IGNORECASE)
        # print(match)
        if match:
            print(match.group(1)) # captures the substring after the pattern
            return response.format(reflect(match.group(1)))
    
    return "Can you tell me more?"

print("ELIZA: Hello! How can I help you today?")
while True:
    user_input = input("You: ")
    if user_input.lower() in ["quit", "exit"]:
        print("ELIZA: Goodbye!")
        break
    print(f"ELIZA: {eliza_response(user_input)}")