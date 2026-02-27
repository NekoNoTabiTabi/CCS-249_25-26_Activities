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

# Store previous inputs to check for repetition
previous_inputs = set()

def eliza_response(user_input):
    """Generates ELIZA-style responses based on input."""

    sarcastic_responses = [
        "Are you deaf?",
        "I'm sensing a pattern here. And by 'pattern,' I mean you repeating yourself.",
        "You know what they say about insanity? It's typing the same thing and expecting a different response.",
        "Groundbreaking. Have you considered professional help for your memory issues?",
        "If repetition is the key to learning, you must be a genius by now.",
        "Is this a test of my patience or your memory?",
    ]

    # Check for repeating input
    input = user_input.strip().lower()
    if input in previous_inputs:
        return random.choice(sarcastic_responses)
    previous_inputs.add(input)

    patterns = [
        (r"I need (.*)", "Why do you need {0}?"),
        (r"Why don’t you (.*)", "Do you really think I don't {0}?"),
        (r"I feel (.*)", "Tell me more about feeling {0}."),

        # a
        (r"I want to know the reasons why I am feeling depressed all the time?", "What do you think is causing your depression?"),
        # b
        (r"I am feeling stressed", "What do you think is causing your stress?"),
        # c
        (r"My feelings towards my crush are invalidated", "Why do you think your feelings are invalidated by your crush?"),
        # d
        (r"You (?:don't|do not) understand me", "What makes you feel that I don't understand you?"),
        #
        (r"I (?:can't| can not) focus on my studies", "What do you think is preventing you from focusing on your studies?")
    ]
    
    for pattern, response in patterns:
        match = re.match(pattern, user_input, re.IGNORECASE)
        # print(match)
        if match:
            if match.lastindex:
                captured_text = match.group(1)
                print(f"Captured text: {captured_text}")
                return response.format(reflect(captured_text))
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