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

def eliza_response(user_input):
    """Generates ELIZA-style responses based on input."""
    patterns = [
        (r"I need (.*)", "Why do you need {0}?"),
        (r"Why don't you (.*)", "Do you really think I don't {0}?"),
        (r"I feel (.*)", "Tell me more about feeling {0}."),
        (r"I want to know (.*)", "What makes you think {0}?"),
        (r"I'?m feeling (.*)", "Why are you feeling {0}?"),
        (r"My feelings towards (.*)", "Tell me more about your feelings towards {0}."),
        (r"You (?:don't|do not) understand (.*)", "Why do you feel I don't understand {0}?"),
        (r"I (?:can't|cannot) focus on (.*)", "What is preventing you from focusing on {0}?")
    ]
    
    for pattern, response in patterns:
        match = re.match(pattern, user_input, re.IGNORECASE)
        # print(match)
        if match:
            print(match.group(1))
            return response.format(reflect(match.group(1)))
    
    return "Can you tell me more?"

print("ELIZA: Hello! How can I help you today?")
previous_questions = []
sarcastic_responses = [
    "Bruhhhhhhhhhh, didn't we just discuss this? Are you testing my memory?",
    "You literally just asked me that (-.-). Maybe try a new question?",
    "Broooo, what are you saying? We just talked about this!",
    "I already answered that. Perhaps you weren't listening?",
    "Repeating yourself won't change my answer, you know (?-_-)."
]

while True:
    user_input = input("You: ")
    if user_input.lower() in ["quit", "exit"]:
        print("ELIZA: Goodbye!")
        break
    
    # Check if the question was asked before
    if user_input.lower() in previous_questions:
        import random
        print(f"ELIZA: {random.choice(sarcastic_responses)}")
    else:
        previous_questions.append(user_input.lower())
        print(f"ELIZA: {eliza_response(user_input)}")