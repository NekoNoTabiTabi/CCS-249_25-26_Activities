# We make the dataset isolated and clean by making it this way


class Dataset:

    def __init__(self, data):
        self.data = data

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


dataset = [
    # Spams
    {"text": "Free money now!!!", "label": "SPAM"},
    {"text": "Lowest price for your meds", "label": "SPAM"},
    {"text": "Win a free iPhone today", "label": "SPAM"},
    {"text": "Get 50% off, limited time!", "label": "SPAM"},
    {"text": "Click here for prizes!", "label": "SPAM"},
    # Hams
    {"text": "Hi mom, how are you?", "label": "HAM"},
    {"text": "Are we still on for dinner?", "label": "HAM"},
    {"text": "Let's catch up tomorrow at the office", "label": "HAM"},
    {"text": "Meeting at 3 PM tomorrow", "label": "HAM"},
    {"text": "Team meeting in the office", "label": "HAM"},
    {"text": "Can you send the report?", "label": "HAM"},
]

dataset = Dataset(dataset)
