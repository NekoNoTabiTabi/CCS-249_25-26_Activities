import re

with open(r'C:\Users\Kyobi Brynat\QB CODEZ\melville-moby_dick.txt', 'r', encoding='utf-8') as file:
    text = file.read()

whales = re.findall(r'\b[wW]hale\b', text, re.IGNORECASE)

print("Number of times 'whale' appears:", len(whales))
