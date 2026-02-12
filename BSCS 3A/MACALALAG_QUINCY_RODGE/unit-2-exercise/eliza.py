# ELIZA implementation in Python
# Example Generated via ChatGPT
import re
import random

# Track previous questions for detecting repeats
previous_questions = []

# Sarcastic responses for repeated questions
sarcastic_responses = [
    "Didn't you just ask me that? My memory isn't that bad, you know.",
    "Oh, we're doing this again? How delightfully repetitive.",
    "I heard you the first time. Were you testing my listening skills?",
    "Déjà vu! Or did you just forget you already asked that?",
    "Ah yes, the classic 'ask the same thing twice' technique. Bold move.",
    "I'm sensing a pattern here... and it's not a very creative one.",
    "You know, repeating yourself won't change my answer. Just saying.",
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
        "me": "you",
    }
    words = fragment.lower().split()
    return " ".join([reflections.get(word, word) for word in words])


def check_repeated_question(user_input):
    """Check if the user is repeating a question and return sarcastic response if so."""
    normalized_input = user_input.lower().strip()
    if normalized_input in previous_questions:
        return random.choice(sarcastic_responses)
    previous_questions.append(normalized_input)
    return None


def eliza_response(user_input):
    """Generates ELIZA-style responses based on input."""
    # BONUS: Check for repeated questions first
    repeated_response = check_repeated_question(user_input)
    if repeated_response:
        return repeated_response

    patterns = [
        (
            r"I want to know the reasons? why I am feeling (.*)",
            "It makes sense to want to understand your feelings of {0}. What do you think might be contributing to this?",
        ),
        (
            r"I am feeling stressed\.?",
            "I'm really sorry you're feeling stressed. Can you share what's been weighing on you lately?",
        ),
        (
            r"My feelings towards (.*) are invalidated\.?",
            "It sounds really hurtful to have your feelings towards {0} not recognized. Would you like to talk about why that might be happening?",
        ),
        (
            r"You don'?t understand me|You do not understand me",
            "I hear you, and I'm sorry you feel misunderstood. Could you help me understand your perspective better?",
        ),
        (
            r"I can'?t focus on my studies|I cannot focus on my studies",
            "It's understandable to struggle with focus sometimes. What do you feel is making it hard to concentrate?",
        ),
        (r"I need (.*)", "Why do you need {0}?"),
        (r"Why don't you (.*)", "Do you really think I don't {0}?"),
        (r"I feel (.*)", "Tell me more about feeling {0}."),
    ]

    for pattern, response in patterns:
        match = re.match(pattern, user_input, re.IGNORECASE)
        if match:
            if match.groups():
                return response.format(reflect(match.group(1)))
            else:
                return response

    return "Can you tell me more?"


print("ELIZA: Hello! How can I help you today?")
while True:
    user_input = input("You: ")
    if user_input.lower() in ["quit", "exit"]:
        print("ELIZA: Goodbye!")
        break
    print(f"ELIZA: {eliza_response(user_input)}")
