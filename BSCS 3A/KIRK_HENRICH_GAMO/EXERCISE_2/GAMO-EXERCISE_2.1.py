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
        "me": "you"
    }
    words = fragment.lower().split()
    return ' '.join([reflections.get(word, word) for word in words])

def eliza_response(user_input, previous_inputs):
    """Generates ELIZA-style responses based on input."""

    # Check for repeated questions
    sarcastic_responses = [
        "Didn't we just talk about this? Are you testing my memory?",
        "Again? Really? I heard you the first time!",
        "You already asked me that. Did you forget my answer?",
        "Oh, we're playing the repetition game now? How fun...",
        "Seriously? The same question again? I'm starting to think you're not listening."
    ]
    
    if user_input.lower() in previous_inputs:
        import random
        return random.choice(sarcastic_responses)
    """Generates ELIZA-style responses based on input."""
    patterns = [
        (r"I need (.*)", "Why do you need {0}?"),
        (r"Why don’t you (.*)", "Do you really think I don't {0}?"),
        (r"I feel (.*)", "Tell me more about feeling {0}."),
        
        # New patterns for Exercise 2
        (r"I want to know.*why.*feeling (\w*)", "Understanding why you're feeling {0} is important. What do you think might be causing this?"),
        (r"I am feeling (\w*)", "What makes you feel {0}?"),
        (r"My feelings towards (.*) are (\w*)", "It sounds like your feelings about {0} are {1}. Can you elaborate?"),
        (r"You (?:don't|do not) understand (\w*)", "I'm trying to understand {0}. Can you help me see it from your perspective?"),
        (r"I (?:can't|cannot) focus on ([^.!?]+)", "What's preventing you from being able to focus on {0}?")
    ]
    
    for pattern, response in patterns:
        match = re.match(pattern, user_input, re.IGNORECASE)
        # print(match)
        if match:
            # Handle multiple capture groups
            reflected_groups = [reflect(group) for group in match.groups()]
            return response.format(*reflected_groups)
    
    return "Can you tell me more?"

previous_inputs = []
while True:
    user_input = input("You: ")
    if user_input.lower() in ["quit", "exit"]:
        print("ELIZA: Goodbye!")
        break
    print(f"ELIZA: {eliza_response(user_input, previous_inputs)}")
    previous_inputs.append(user_input.lower())