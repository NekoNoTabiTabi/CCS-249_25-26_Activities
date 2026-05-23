# Unit 2 – Exercise 2
# Tasks: 2a, 2b, 2c
# Name: Ryan Cabalfin
# Section: CS 3A
import re

# ===============================
# TASK 2a
# ===============================
def task_2a():
    text = """Alice was beginning to get very tired of sitting by her sister on the bank,
and of having nothing to do. Once or twice she had peeped into the book
her sister was reading, but it had no pictures or conversations in it, "and
what is the use of a book," thought Alice, "without pictures or
conversations?"""

    pattern = r"\b[A-Z][a-zA-Z]*\b"
    uppercase_words = re.findall(pattern, text)

    print("=== Task 2a Output ===")
    print(uppercase_words)


# ===============================
# TASK 2b
# ===============================
def task_2b(input_filename="melville-moby_dick.txt", output_filename="moby_dick_leviathan.txt"):
    try:
        with open(input_filename, "r", encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError:
        print(f"ERROR: Cannot find '{input_filename}'. Put it in the same folder as this .py file.")
        return
    except UnicodeDecodeError:
        # fallback encoding if utf-8 fails
        with open(input_filename, "r", encoding="latin-1") as f:
            text = f.read()

    pattern = r"\bwhales?\b"  # matches whale or whales
    matches = re.findall(pattern, text, flags=re.IGNORECASE)

    replaced_text = re.sub(pattern, "leviathan", text, count=10, flags=re.IGNORECASE)

    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(replaced_text)

    print("=== Task 2b Output ===")
    print(f"Total matches found (all Whale/whales variants): {len(matches)}")
    print(f"Replaced first 10 occurrences with 'leviathan'. Saved to: {output_filename}")


# ===============================
# TASK 2c
# ===============================
def task_2c():
    try:
        import nltk
        from nltk.corpus import webtext
    except ImportError:
        print("ERROR: NLTK is not installed. Run: pip install nltk")
        return

    # Download webtext if not present (safe to run even if already downloaded)
    try:
        nltk.data.find("corpora/webtext")
    except LookupError:
        nltk.download("webtext")

    raw_text = webtext.raw("pirates.txt")
    lines = raw_text.splitlines()

    # Match speaker labels like:
    # JACK: ...
    # JACK SPARROW: ...
    # Jack: ...
    # Jack Sparrow: ...
    pattern = r"^\s*(?:JACK(?:\s+SPARROW)?|Jack(?:\s+Sparrow)?)\s*:\s*.+"

    jack_lines = [line for line in lines if re.match(pattern, line)]

    print("=== Task 2c Output ===")
    print(f"Total Jack Sparrow lines found: {len(jack_lines)}\n")
    for line in jack_lines:
        print(line)

# ===============================
# MAIN MENU
# ===============================
def main():
    print("Choose a task to run:")
    print("1 - Task 2a (Uppercase words)")
    print("2 - Task 2b (Replace whale -> leviathan)")
    print("3 - Task 2c (Jack Sparrow lines)")
    choice = input("Enter choice (1/2/3): ").strip()

    if choice == "1":
        task_2a()
    elif choice == "2":
        task_2b()
    elif choice == "3":
        task_2c()
    else:
        print("Invalid choice.")

if __name__ == "__main__":
    main()