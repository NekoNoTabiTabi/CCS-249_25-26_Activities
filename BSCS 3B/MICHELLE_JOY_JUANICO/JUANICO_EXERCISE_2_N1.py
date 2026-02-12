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
        (r"Why don’t you (.*)", "Do you really think I don't {0}?"),
        (r"I feel (.*)", "Tell me more about feeling {0}."),
        (r"I want to know (.*)", "Why is it important for you to know {0}?"),
        (r"I am feeling (.*)", "Is being {0} important?"),
        (r"My feelings towards my crush are (.*)", "What is the reason why you're feeling {0}?"),
        (r"You (?:don't|do not) understand (.*)", "How can you tell that I don't understand {0}?"),
        (r"I (?:can't|cannot) (.*)", "Why is it that you can't {0}?")
    ]
    
    for pattern, response in patterns:
        match = re.match(pattern, user_input, re.IGNORECASE)
        # print(match)
        if match:
            print(match.group(1)) # captures the substring after the pattern
            return response.format(reflect(match.group(1)))
    
    return "Can you tell me more?"

print("ELIZA: Hello! How can I help you today?")
previous_questions = []
while True:
    user_input = input("You: ")
    if user_input.lower() in ["quit", "exit"]:
        print("ELIZA: Goodbye!")
        break
    
    # Check if the question was already asked
    if user_input in previous_questions:
        print("ELIZA: Really? You're so clever, that's just what I needed today!")
    else:
        previous_questions.append(user_input)
        print(f"ELIZA: {eliza_response(user_input)}")