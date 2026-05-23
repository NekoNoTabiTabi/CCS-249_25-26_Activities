# JULLIAN BILAN | BSCS 3A | Exercise 2.1

# 1.	Updating ELIZA. Modify the ELIZA python implementation shared to the class by performing the following:

# (10 points each) Add 5 RegEx patterns to capture:

# a.	I want to know the reasons why I am feeling depressed all the time.
# b.	I am feeling stressed.
# c.	My feelings towards my crush are invalidated.
# d.	You don’t understand me OR You do not understand me.
# e.	I can’t focus on my studies OR I cannot focus on my studies.

import re
import random

# Track previous user inputs for detecting repeated questions
previous_inputs = []

# Sarcastic responses for repeated questions
sarcastic_responses = [
    "Liwat liwat kaya imo haw",
    "Baw uli-anon",
    "Nanaman?",
    "huo, bati taka",
    "ano abi mo sakon bungol?"
]

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
    patterns = [
        (r"I need (.*)", "Why do you need {0}?"),
        (r"Why don’t you (.*)", "Do you really think I don't {0}?"),
        (r"I feel (.*)", "Tell me more about feeling {0}."),
        # a. I want to know the reasons why I am feeling depressed all the time.
        (r"I want to know the reasons why I am feeling (.*) all the time", "What do you think is causing you to feel {0} all the time?"),
        # b. I am feeling stressed.
        (r"I am feeling (.*)", "Why do you think you are feeling {0}?"),
        # c. My feelings towards my crush are invalidated.
        (r"My feelings towards (.*) are (.*)", "Why do you feel your feelings towards {0} are {1}?"),
        # d. You don't understand me OR You do not understand me.
        (r"You do(?:n't| not) understand me", "What makes you think I don't understand you?"),
        # e. I can't focus on my studies OR I cannot focus on my studies.
        (r"I can(?:n't|not) focus on (.*)", "What do you think is preventing you from focusing on {0}?")
    ]
    
    for pattern, response in patterns:
        match = re.match(pattern, user_input, re.IGNORECASE)
        if match:
            # Get all captured groups and reflect them
            groups = [reflect(g) if g else "" for g in match.groups()]
            return response.format(*groups)
    
    return "Can you tell me more?"

print("ELIZA: Hello! How can I help you today?")
while True:
    user_input = input("You: ")
    if user_input.lower() in ["quit", "exit"]:
        print("ELIZA: Goodbye!")
        break
    
    # BONUS: Check if user is repeating the same question
    if user_input.lower() in [prev.lower() for prev in previous_inputs]:
        print(f"ELIZA: {random.choice(sarcastic_responses)}")
    else:
        print(f"ELIZA: {eliza_response(user_input)}")
    
    # Add input to history
    previous_inputs.append(user_input)