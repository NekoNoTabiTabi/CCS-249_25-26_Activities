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

# Track previous user inputs for the bonus feature
previous_inputs = []

def eliza_response(user_input):
    """Generates ELIZA-style responses based on input."""
    
    # BONUS: Check if the user is repeating the same question
    if user_input.lower() in previous_inputs:
        sarcastic_responses = [
            "Didn't we just talk about this? Let's try something new.",
            "I heard you the first time. Same question, same answer - try asking something different.",
            "Are you testing my memory? We've covered this already.",
            "Repeating yourself won't change my answer. What else is on your mind?",
            "Oh, we're doing reruns now? I prefer fresh content."
        ]
        return random.choice(sarcastic_responses)
    
    # Add to previous inputs for tracking
    previous_inputs.append(user_input.lower())
    
    patterns = [
        (r"I need (.*)", "Why do you need {0}?"),
        (r"Why don't you (.*)", "Do you really think I don't {0}?"),
        (r"I feel (.*)", "Tell me more about feeling {0}."),
        # New Pattern (a): I want to know the reasons why I am feeling depressed all the time
        (r"I want to know the reasons? why I am feeling depressed( all the time)?", 
         "Depression can have many causes. What do you think might be contributing to your feeling depressed?"),
        # New Pattern (b): I am feeling stressed
        (r"I am feeling stressed", 
         "What is making you feel stressed? Can you tell me more about the sources of your stress?"),
        # New Pattern (c): My feelings towards my crush are invalidated
        (r"My feelings towards my crush are invalidated", 
         "Why do you feel your feelings are being invalidated? Have you tried expressing how important these feelings are to you?"),
        # New Pattern (d): You don't understand me OR You do not understand me
        (r"You (?:don't|do not) understand me", 
         "I'm trying my best to understand you. Can you help me understand better by explaining more?"),
        # New Pattern (e): I can't focus on my studies OR I cannot focus on my studies
        (r"I (?:can't|cannot) focus on my studies", 
         "What do you think is preventing you from focusing on your studies? Are there specific distractions or concerns?")
    ]
    
    for pattern, response in patterns:
        match = re.match(pattern, user_input, re.IGNORECASE)
        if match:
            # Check if there's a captured group
            if match.lastindex:
                return response.format(reflect(match.group(1)))
            else:
                return response
    
    return "Can you tell me more?"

# Main conversation loop
if __name__ == "__main__":
    print("ELIZA: Hello! How can I help you today?")
    print("(Type 'quit' or 'exit' to end the conversation)\n")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["quit", "exit"]:
            print("ELIZA: Goodbye!")
            break
        print(f"ELIZA: {eliza_response(user_input)}\n")
