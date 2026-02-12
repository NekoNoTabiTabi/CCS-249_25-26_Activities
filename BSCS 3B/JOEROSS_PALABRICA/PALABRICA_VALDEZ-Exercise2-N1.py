# ELIZA implementation in Python
# Example Generated via ChatGPT
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


def eliza_response(user_input, previous_inputs):
    """Generates ELIZA-style responses based on input."""
    # BONUS
    if user_input.lower() in previous_inputs:
        sarcastic_responses = [
            "Didn't we just talk about this? Are you testing my memory?",
            "I heard you the first time. Why are you asking again?",
            "Really? The same question again? Do you expect a different answer?",
            "You already asked me that. Is there something else on your mind?",
            "I'm starting to think you're not listening to my responses...",
            "We've been through this already. Let's move forward, shall we?"
        ]
        import random
        return random.choice(sarcastic_responses)
    
    patterns = [
        (r"I need (.*)", "Why do you need {0}?"),
        (r"Why don't you (.*)", "Do you really think I don't {0}?"),
        (r"I feel (.*)", "Tell me more about feeling {0}."),
        (r"I want to know the reasons why I am feeling (.*)", "What do you think might be behind these feelings of {0}?"),
        (r"I am feeling (.*)", "Can you tell me more about when you started feeling {0}?"),
        (r"You (?:don't|do not) understand me", "Why do you feel that I don't understand you?"),
        (r"I (?:can't|cannot) focus on my studies", "What is making it hard for you to focus on your studies?"),
        
    ]

    for pattern, response in patterns:
        match = re.match(pattern, user_input, re.IGNORECASE)
        if match:
            if match.lastindex:
                return response.format(reflect(match.group(1)))
            else:
                return response

    return "Can you tell me more?"


print("ELIZA: Hello! How can I help you today?")
previous_inputs = []
while True:
    user_input = input("You: ")
    if user_input.lower() in ["quit", "exit"]:
        print("ELIZA: Goodbye!")
        break
    print(f"ELIZA: {eliza_response(user_input, previous_inputs)}")
    previous_inputs.append(user_input.lower())

