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

def eliza_response(user_input, history):
    """Generates ELIZA-style responses based on input."""
    if user_input.lower() in [h.lower() for h in history]:
        sarcastic_responses = [
            "Woooow..., never heard that one before... oh wait, I just did.",
            "Are we stuck in a time loop? I'm getting déjà vu here.",
            "Ayus no. You know what would be more interesting? lain naman.",
            "Are you serious? again?",
            "Pa ulit-ulit na naman tayo, ano ba? wala ba ibang laman ang utak mo?",
            "你是认真的?!!",
            "Boring mo naman kausap, iban naman b?"
        ]
        return random.choice(sarcastic_responses)
    
    patterns = [        
        (r"I want to know.*why (.*)", "That's an interesting question. What do you think might be causing {0}?"),
        (r"I am feelings? (.*)", "It sounds like {0} is weighing on you. How long have you been experiencing this?"),
        (r"My feelings towards (.*) are invalidated", "I hear you there. Your feelings about {0} are valid. Can you tell me more about what happened?"),
        (r"you don'?t understand (.*)", "I want to understand {0}. Can you help me see your perspective?"),
        (r"I can'?t focus on (.*)", "Difficulty focusing on {0} can be tricky. What's been distracting you?"),        (r"I need (.*)", "Why do you need {0}?"),
        (r"Why don’t you (.*)", "Do you really think I don't {0}?"),
        (r"I feel (.*)", "Tell me more about feeling {0}."),
        (r"I am feeling (.*)", "These could be the reasons why you are feeling {0}.")
    ]
    
    for pattern, response in patterns:
        match = re.search(pattern, user_input, re.IGNORECASE)
        # print(match)
        if match:
            print(match.group(1)) # captures the substring after the pattern
            return response.format(reflect(match.group(1)))
    
    return "Can you tell me more?"

print("ELIZA: Hello! How can I help you today?")
history = []
while True:
    user_input = input("You: ")
    if user_input.lower() in ["quit", "exit"]:
        print("ELIZA: Goodbye!")
        break
    response = eliza_response(user_input, history)
    history.append(user_input)
    print(f"ELIZA: {response}")