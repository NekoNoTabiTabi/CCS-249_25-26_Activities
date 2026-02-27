# 1. Updating ELIZA
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
    return ' '.join(reflections.get(word, word) for word in words)

def eliza_response(user_input):
    """Generates ELIZA-style responses based on input."""
    patterns = [
        (r"I need (.*)", "Why do you need {0}?"),
        (r"Why don['’]t you (.*)", "Do you really think I don't {0}?"),
        (r"I feel (.*)", "Tell me more about feeling {0}."),

        # Required patterns
        (r"I want to know the reasons why I am feeling depressed all the time\.?",
         "Why do you want to know the reasons for feeling depressed all the time?"),

        (r"I am feeling stressed\.?",
         "Tell me more about why you're feeling stressed."),

        (r"My feelings towards my crush are invalidated\.?",
         "Why do you think your feelings towards your crush are invalidated?"),

        (r"You (don['’]t|do not) understand me\.?",
         "Why do you think I don't understand you?"),

        (r"I (can['’]t|cannot) focus on my studies\.?",
         "What is making it difficult for you to focus on your studies?")
    ]

    for pattern, response in patterns:
        match = re.match(pattern, user_input, re.IGNORECASE)
        if match:
            if match.groups():
                return response.format(reflect(match.group(1)))
            return response

    return "Can you tell me more?"

# Track previous questions
previous_questions = []

sarcastic_responses = [
    "Yes, you've already said that. Let's try something new.",
    "We seem to be going in circles, don't we?",
    "Repeating the question won't change my answer.",
    "I remember. Do you remember what I said?",
    "Still thinking about that, huh?"
]

print("ELIZA: Hello! How can I help you today?")
while True:
    user_input = input("You: ")

    if user_input.lower() in ["quit", "exit"]:
        print("ELIZA: Goodbye!")
        break

    # Check for repeated question
    if user_input.lower() in [q.lower() for q in previous_questions]:
        print("ELIZA:", random.choice(sarcastic_responses))
    else:
        response = eliza_response(user_input)
        print(f"ELIZA: {response}")
        previous_questions.append(user_input)
