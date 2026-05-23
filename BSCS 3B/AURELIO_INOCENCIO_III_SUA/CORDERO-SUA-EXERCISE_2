# ELIZA implementation in Python
# Example Generated via ChatGPT
import re
import os
import nltk
from nltk.corpus import webtext

nltk.download('webtext')


# # === PIRATES ===
pirates_text = webtext.raw('pirates.txt')

lines = pirates_text.split('\n')

# Extract all Jack Sparrow dialogue using Option 1
jack_sparrow_dialogue = [re.match(r"^JACK SPARROW:\s*(.*)", line).group(1) 
                         for line in lines 
                         if re.match(r"^JACK SPARROW:\s*(.*)", line)]

for dialogue in jack_sparrow_dialogue:
    print(dialogue)


# # === ALICE ===
alice_text = "Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do. Once or twice she had peeped into the book her sister was reading, but it had no pictures or conversations in it, 'and what is the use of a book,' thought Alice 'without pictures or conversations?'"

uppercase_alice = re.findall(r"\b[A-Za-z]*[A-Z][A-Za-z]*\b", alice_text)
print(uppercase_alice)

# === MOBY DICK ===
script_dir = os.path.dirname(os.path.abspath(__file__))
moby_file = os.path.join(script_dir, 'melville-moby_dick.txt')

with open(moby_file, 'r', encoding='utf-8') as file:
    moby_text = file.read()

whale_matches = re.findall(r"\bwhale\b", moby_text)
print(f"Found {len(whale_matches)} occurrences of 'whale'")
print(whale_matches[:5])  



# def reflect(fragment):
#     """Reflects user input to make responses more natural."""
#     reflections = {
#         "am": "are",
#         "was": "were",
#         "i": "you",
#         "i'd": "you would",
#         "i've": "you have",
#         "i'll": "you will",
#         "my": "your",
#         "are": "am",
#         "you've": "I have",
#         "you'll": "I will",
#         "your": "my",
#         "yours": "mine",
#         "you": "me",
#         "me": "you"
#     }
#     words = fragment.lower().split()
#     return ' '.join([reflections.get(word, word) for word in words])

# def eliza_response(user_input):
#     """Generates ELIZA-style responses based on input."""

#     # Sarcasm if user repeats the same input 
#     # if last_input and user_input.strip().lower() == last_input.strip().lower(): 
#     # sarcastic_responses = [ 
#     # "Didn't you just say that?", 
#     # "You must really like repeating yourself.", 
#     # "Trying to drill it into my circuits, huh?", 
#     # "I heard you the first time.", "Echo... echo... echo..." 
#     # ] 
#     # return random.choice(sarcastic_responses)

#     patterns = [
#         (r"I need (.*)", "Why do you need {0}?"),
#         (r"Why don’t you (.*)", "Do you really think I don't {0}?"),
#         (r"I feel (.*)", "Tell me more about feeling {0}."),
#         (r"I want to know the reasons why I am feeling (.*)", "Why do you think you are feeling {0}?"), 
#         (r"I am feeling stressed(.*)", "Stress can affect many aspects of life. What do you think is causing your stress{0}?"),

#         (r"My feelings towards my crush are invalidated(.*)", "It sounds painful to feel invalidated in your feelings. Can you share more about that{0}?"),
#         (r"You don't understand me|You do not understand me(.*)", "I may not fully understand yet, but I'd like to. What makes you feel misunderstood{0}?"),
#         (r"I can't focus on my studies|I cannot focus on my studies(.*)", "Difficulty focusing can be stressful. What do you think is affecting your concentration{0}?"),

#         (r"I am feeling depressed(.*)", "Depression can be overwhelming. What do you think contributes to feeling depressed{0}?"),  
#         (r"I want to understand why I feel (.*)", "Understanding feelings of {0} can be important. What do you think lies behind them?"), 
#         (r"I keep feeling (.*) all the time", "Why do you think you keep feeling {0} so often?")
#     ]
    
#     for pattern, response in patterns:
#         match = re.match(pattern, user_input, re.IGNORECASE) #re.search - If you want to find pattern anywhere in the string
#         # print(match)
#         if match:
#             # print(match.group(1)) # captures the substring after the pattern
#             # last_input = user_input -- for sarcasm feature
#             return response.format(reflect(match.group(1)))
    
#     return "Can you tell me more?"

# print("ELIZA: Hello! How can I help you today?")
# while True:
#     user_input = input("You: ")
#     if user_input.lower() in ["quit", "exit"]:
#         print("ELIZA: Goodbye!")
#         break
#     print(f"ELIZA: {eliza_response(user_input)}")