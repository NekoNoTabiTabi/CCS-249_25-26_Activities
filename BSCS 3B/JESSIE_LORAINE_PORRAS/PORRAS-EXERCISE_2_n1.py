# ELIZA implementation in Python
# Example Generated via ChatGPT

#1. 1.	Updating ELIZA
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

    fragment = re.sub(r"[^\w\s]", "", fragment.lower())
    words = fragment.split()
    return ' '.join([reflections.get(word, word) for word in words])

prev_input = []

def eliza_response(user_input):
    """Generates ELIZA-style responses based on input."""
    normalized = normalize(user_input)

    if normalized in prev_input:
        return random.choice(sarcastic_responses)
    
    prev_input.append(normalized)

    patterns = [
        (r"I need (.*)", "Why do you need {0}?"),
        (r"Why don’t you (.*)", "Do you really think I don't {0}?"),
        (r"I feel (.*)", "Tell me more about feeling {0}."),

        (r"I want to know (.*)", "Why do you want to know {0}?"),
        (r"Can you tell me.*reason.*why I (?:am|feel) (.*)", "What do you think might be the reason you feel that way?"),
        (r"Is there a (reason|cause).*I (?:am|feel) (.*)", "Why do you want to know the {0}?"),
        (r"What is making me (.*)", "What do you think is causing that?"),
        (r"Why do I feel (.*)", "Did something happened that made you feel {0}?"),
        
        (r"I am feeling (.*)", "Why do you think you are feeling that way?"),
        (r"I(?:'m|am).*(?:feeling|feeling like|like) (.*)", "I'm sorry you're feeling this way. Can you tell me more about it?"),
        (r"What can I do to (.*)", "What have you tried so far to {0}?"),
        #(r"I feel (.*)", "What do you think is causing you to feel that way?"),
        #(r"Why do I feel (.*)", "Why do you think you feel that way?"),

        (r"My feelings .* crush are (.*)", "Why do you think that your feelings are {0}?"),

        (r"You (?:don't|do not) (.*)", "I'm sorry. Why do you think I don't {0}?"),

        (r"I (?:can't|cannot) (.*)", "Please tell me more about why you can't {0}."),

    ]
    
    for pattern, response in patterns:
        match = re.match(pattern, user_input, re.IGNORECASE)
        # print(match)
        if match:
            print(match.group(1)) # captures the substring after the pattern
            return response.format(reflect(match.group(1)))
    
    return "Can you tell me more?"

def normalize(text):
    text = re.sub(r"[^\w\s]", "", text.lower())
    return re.sub(r'\s+', ' ', text.lower()).strip()

sarcastic_responses = [
    "Oh, really? Tell me more about that.",
    "You've already asked me that. What do you think the answer is?",
    "I'm sorry, I didn't quite catch that. Could you rephrase it?",
    "Interesting. Are you hoping for a different answer this time?",
    "We appear to be going in circles. Let's try a new topic.",
]

print("ELIZA: Hello! How can I help you today?")
while True:
    user_input = input("You: ")
    if user_input.lower() in ["quit", "exit"]:
        print("ELIZA: Goodbye!")
        break
    print(f"ELIZA: {eliza_response(user_input)}")