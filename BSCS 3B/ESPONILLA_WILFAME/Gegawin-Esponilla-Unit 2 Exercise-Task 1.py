# ELIZA implementation in Python
# Example Generated via ChatGPT
import re

# prev_question
previous_inputs = []

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
        "is": "being"
    }
    words = fragment.lower().split()
    return ' '.join([reflections.get(word, word) for word in words])

# normalize input for similarity checking
def normalize(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)   # remove punctuation
    return set(text.split())


def eliza_response(user_input):
    """Generates ELIZA-style responses based on input."""

    # similarity detection
    normalized_input = normalize(user_input)
    for prev in previous_inputs:
        common_words = normalized_input.intersection(prev)
        if len(common_words) >= 3:  # threshold for "similar"
            return "You're asking the same thing again, Did you expect me to give a different response? Interesting choice."

    # store normalized input
    previous_inputs.append(normalized_input)

    patterns = [
    # Existing patterns
    (r"I need (.*)", "Why do you need {0}?"),
    (r"Why don’t you (.*)", "Do you really think I don't {0}?"),
    (r"I feel (.*)", "Tell me more about feeling {0}."),

    # New patterns
    (r"I want to know (?:the )?reasons why I am feeling(.*)",
    "What do you think causes you to feel {0}?"),
    (r"I am feeling (.*)", 
     "How long have you been feeling {0}?"),
    (r"My feelings towards (.*)", 
     "Can you elaborate on your feelings towards {0}?"),
    (r"You (?:do not|don't) understand me(.*)", 
     "What makes you feel misunderstood?"),
    (r"I (?:cannot|can't) focus on my (.*)",
    "What do you think is making it hard to focus on your {0}?")
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