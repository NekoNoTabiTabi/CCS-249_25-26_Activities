import re

# Store last user input for repetition check
last_input = None

def reflect(fragment):
    reflections = {
        "am": "are",
        "was": "were",
        "i": "you",
        "i'm": "you are",
        "my": "your",
        "me": "you",
        "you": "me",
        "your": "my"
    }
    words = fragment.lower().split()
    return " ".join([reflections.get(word, word) for word in words])

def eliza_response(user_input):
    global last_input

    # BONUS: sarcastic reply if user repeats the same question
    if user_input.lower() == last_input:
        return "Yes, you already asked that. Maybe the answer didn’t magically change?"

    last_input = user_input.lower()

    patterns = [
        # a. Feeling depressed all the time
        (r"I want to know the reasons why I am feeling depressed all the time",
         "What do you think might be causing you to feel depressed all the time?"),

        # b. Feeling stressed
        (r"I am feeling stressed",
         "What do you think is making you feel stressed?"),

        # c. Feelings toward crush invalidated
        (r"My feelings towards my crush are invalidated",
         "How does it affect you when your feelings toward your crush are invalidated?"),

        # d. You don’t / do not understand me
        (r"You (don’t|do not) understand me",
         "Why do you feel that I don’t understand you?"),

        # e. Can’t / cannot focus on studies
        (r"I (can’t|cannot) focus on my studies",
         "What do you think is stopping you from focusing on your studies?")
    ]

    for pattern, response in patterns:
        match = re.match(pattern, user_input, re.IGNORECASE)
        if match:
            return response

    return "Can you tell me more?"

print("ELIZA: Hello! How can I help you today?")
while True:
    user_input = input("You: ")
    if user_input.lower() in ["quit", "exit"]:
        print("ELIZA: Goodbye!")
        break
    print("ELIZA:", eliza_response(user_input))