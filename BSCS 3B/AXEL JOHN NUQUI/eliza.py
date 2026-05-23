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

last_input = None

def eliza_response(user_input):
    global last_input
    
    repeated_ques = user_input.lower().strip()

    if repeated_ques == last_input:
        return "Yeye, ulet ulet?"

    last_input = repeated_ques
    
    """Generates ELIZA-style responses based on input."""
    patterns = [
        (r"I need (.*)", "Why do you need {0}?"),
        (r"Why don’t you (.*)", "Do you really think I don't {0}?"),
        (r"I feel (.*)", "Tell me more about feeling {0}."),
        (r"I want to know the reasons why I am feeling (.*)", "Why do you think you're feeling {0}?"),
        (r"I am feeling (.*)", "When did you start feeling {0}?"),
        (r"My feelings towards my crush are (.*)", "Why do you think your makes you feel {0}?")
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