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


prev_input = None


def eliza_response(user_input):

    global prev_input

    """Generates ELIZA-style responses based on input."""
    patterns = [
        #  (.*) basically means capture anything and store group 0
        (r"I need (.*)", "Why do you need {0}?"),
        (r"Why don’t you (.*)", "Do you really think I don't {0}?"),
        (r"I feel (.*)", "Tell me more about feeling {0}."),
        (r"I want to know the (?:reasons|reason) why I am (.*)", "Sorry, but I dont know too. Maybe it is your habits that makes you {0}"),
        (r"I am feeling (.*)", "Why do you think you're feeling {0}?"),
        (r"My feelings toward my crush are (.*)", "Why do you think your crush {0} your feelings?"),
        (r"You (?:don't|do not|dont) (.*)", "Why do you think that?"),
        (r"I (?:can't|can not|cant) (.*)", "Why do you think you can not {0}?")
    ]
    

    if prev_input == user_input:
            prev_input = user_input
            return "How many times do you need to repeat yourself????"

    for pattern, response in patterns:
        match = re.match(pattern, user_input, re.IGNORECASE)

        # print(match)  
        if match:
            prev_input = user_input # also check for repetition even with a match
            # print(match.group(1)) # captures the substring after the pattern
            return response.format(reflect(match.group(1)))

    # if user input is not prev input, and no match, still store the user input as prev input for next iteration
    prev_input = user_input
    return "Can you tell me more?"




print("ELIZA: Hello! How can I help you today?")
while True:
    user_input = input("You: ")
    if user_input.lower() in ["quit", "exit"]:
        print("ELIZA: Goodbye!")
        break
    print(f"ELIZA: {eliza_response(user_input)}")