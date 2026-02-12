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

        (
            r"I want to know the reasons why I am feeling depressed all the time",
            "What do you think might be contributing to these feelings of depression?"
        ),

        (
            r"I am feeling stressed",
            "What do you think is causing this stress?"
        ),

        (
            r"My feelings towards my crush are invalidated",
            "How does it feel when your emotions about your crush are invalidated?"
        ),

        (
            r"You (don’t|do not) understand me",
            "What makes you feel that I don't understand you?"
        ),

        (
            r"I (can’t|cannot) focus on my studies",
            "What do you think is making it hard for you to focus on your studies?"
        ),

        (r"I need (.*)", "Why do you need {0}?"),
        (r"Why don’t you (.*)", "Do you really think I don't {0}?"),
        (r"I feel (.*)", "Tell me more about feeling {0}.")
    ]
    
    for pattern, response in patterns:
        match = re.match(pattern, user_input, re.IGNORECASE)
        if match:
            if match.lastindex:
                return response.format(reflect(match.group(match.lastindex)))
            return response
    
    return "Can you tell me more?"

print("ELIZA: Hello! How can I help you today?")

previous_input = ""  #Memory 

while True:
    user_input = input("You: ")

    if user_input.lower() in ["quit", "exit"]:
        print("ELIZA: Goodbye!")
        break

    if user_input.lower() == previous_input.lower():
        print("ELIZA: You already asked that. Were you expecting a different answer this time?")
    else:
        print(f"ELIZA: {eliza_response(user_input)}")

    previous_input = user_input
