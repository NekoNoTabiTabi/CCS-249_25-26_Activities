# ELIZA implementation in Python
# Example Generated via ChatGPT
import re
import random

# Track previous user inputs for detecting repetition (BONUS)
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
        "me": "you"
    }
    words = fragment.lower().split()
    return ' '.join([reflections.get(word, word) for word in words])

def eliza_response(user_input):
    """Generates ELIZA-style responses based on input."""
    # BONUS: Check if user is repeating the same question
    normalized_input = user_input.lower().strip()
    if normalized_input in previous_inputs:
        sarcastic_responses = [
            "Didn't we just talk about this? Are you testing my memory?",
            "You already asked me that. Do you think I'll give you a different answer?",
            "Really? The same question again? Maybe you need to work on your listening skills.",
            "I answered this already. Are you not satisfied with my response, or just forgetful?",
            "Here we go again... Repeating questions won't change reality, you know."
        ]
        return random.choice(sarcastic_responses)
    
    # Track this input for future repetition detection
    previous_inputs.append(normalized_input)
    
    patterns = [
        # Original patterns
        (r"I need (.*)", "Why do you need {0}?"),
        (r"Why don't you (.*)", "Do you really think I don't {0}?"),
        (r"I feel (.*)", "Tell me more about feeling {0}."),
        
        # New patterns (a-e)
        # a. Pattern for depression/feeling depressed
        (r".*(?:depressed|depression).*", "Depression is serious. What do you think might be contributing to these feelings of being depressed?"),
        
        # b. Pattern for feeling stressed
        (r"I am feeling (stressed|anxious|overwhelmed|nervous)", "When you're feeling {0}, what helps you cope? Have you identified any specific triggers?"),
        
        # c. Pattern for invalidated feelings towards crush
        (r"My feelings?.* (?:towards|for|about) (?:my )?(?:crush|someone).*(?:invalidated|ignored|dismissed|rejected)", "Having your feelings invalidated can be painful. Why do you think your feelings towards this person are being dismissed?"),
        
        # d. Pattern for 'you don't understand me'
        (r"You (?:don't|do not|dont) understand (?:me)?", "I want to understand you better. What makes you feel that I don't understand you?"),
        
        # e. Pattern for can't focus on studies
        (r"I (?:can't|cannot|cant) focus (?:on )?(?:my )?(?:studies|work|school|studying)", "Difficulty focusing can be frustrating. What do you think is preventing you from concentrating on your studies?")
    ]
    
    for pattern, response in patterns:
        match = re.match(pattern, user_input, re.IGNORECASE)
        # print(match)
        if match:
            # Check if pattern has a capture group
            if match.lastindex and match.lastindex >= 1:
                print(match.group(1)) # captures the substring after the pattern
                return response.format(reflect(match.group(1)))
            else:
                # No capture group, return response as-is
                return response
    
    return "Can you tell me more?"

print("ELIZA: Hello! How can I help you today?")
while True:
    user_input = input("You: ")
    if user_input.lower() in ["quit", "exit"]:
        print("ELIZA: Goodbye!")
        break
    print(f"ELIZA: {eliza_response(user_input)}")