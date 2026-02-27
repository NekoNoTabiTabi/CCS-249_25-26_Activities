# ELIZA implementation in Python
# Example Generated via ChatGPT
# Updated with 5 new patterns and bonus repetition detection
import re

# Track previous user inputs for detecting repetition (BONUS feature)
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
    
    # BONUS: Check if user is repeating a previous question
    normalized_input = user_input.lower().strip()
    if normalized_input in previous_inputs:
        sarcastic_responses = [
            "Oh, are we asking the same question again? How interesting...",
            "Didn't we just discuss this? Perhaps you weren't listening to yourself.",
            "Repetition is the hallmark of... well, repetition. Try something new?",
            "I already answered that. Are you testing my memory or yours?",
            "Asking the same thing twice won't magically change my answer, you know."
        ]
        # Rotate through sarcastic responses based on number of repetitions
        return sarcastic_responses[len([x for x in previous_inputs if x == normalized_input]) % len(sarcastic_responses)]
    
    # Add to history
    previous_inputs.append(normalized_input)
    
    patterns = [
        # Pattern a: I want to know the reasons why I am feeling depressed all the time.
        (r".*want to know the reasons? why (?:I am|I'm) feeling depressed.*", 
         "Depression can have many causes. What do you think might be contributing to your feelings of being depressed?"),
        
        # Pattern b: I am feeling stressed.
        (r".*(?:I am|I'm) feeling stressed.*", 
         "Stress affects us all. What specifically is making you feel stressed right now?"),
        
        # Pattern c: My feelings towards my crush are invalidated.
        (r".*feelings towards my crush are invalidated.*", 
         "Having your feelings invalidated can be painful. Why do you think your feelings towards your crush aren't being acknowledged?"),
        
        # Pattern d: You don't understand me OR You do not understand me.
        (r".*you (?:don't|do not) understand me.*", 
         "I want to understand you better. What is it that you feel I'm not grasping about your situation?"),
        
        # Pattern e: I can't focus on my studies OR I cannot focus on my studies.
        (r".*I (?:can't|cannot) focus on my studies.*", 
         "Difficulty focusing on studies is common. What do you think is preventing you from concentrating?"),
        
        # Original patterns
        (r"I need (.*)", "Why do you need {0}?"),
        (r"Why don't you (.*)", "Do you really think I don't {0}?"),
        (r"I feel (.*)", "Tell me more about feeling {0}.")
    ]
    
    for pattern, response in patterns:
        match = re.match(pattern, user_input, re.IGNORECASE)
        # print(match)
        if match:
            # Check if pattern has capture group
            if match.lastindex:
                print(match.group(1)) # captures the substring after the pattern
                return response.format(reflect(match.group(1)))
            else:
                return response
    
    return "Can you tell me more?"

print("ELIZA: Hello! How can I help you today?")
while True:
    user_input = input("You: ")
    if user_input.lower() in ["quit", "exit"]:
        print("ELIZA: Goodbye!")
        break
    print(f"ELIZA: {eliza_response(user_input)}")
