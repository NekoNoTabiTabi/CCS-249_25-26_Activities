from datasets import load_dataset, Dataset, concatenate_datasets
import json

local_dataset = load_dataset(
    "json", data_files="./dataset/wvsu_dataset.jsonl", split="train"
)

# STREAM 50 items from Tanaos (Greetings/Thanks)
tanaos_stream = load_dataset(
    "tanaos/synthetic-intent-classifier-dataset-v1", split="train", streaming=True
)

# We only want label 0 (greeting) and label 2 (thank_you)
# We'll re-map them to Label 4 (General/Politeness) for the model
general_items = []
for item in tanaos_stream:
    if item["labels"] in [0, 2]:  # Greeting or Thank You
        general_items.append({"text": item["text"], "label": 4})
        print(item)
    if len(general_items) >= 50:
        break


general_dataset = Dataset.from_list(general_items)

print(f"Local Columns: {local_dataset.column_names}")
print(f"General Columns: {general_dataset.column_names}")

# Combine them
final_train_dataset = concatenate_datasets([local_dataset, general_dataset])
final_train_dataset = final_train_dataset.shuffle(seed=42)

# PROOF CHECK: Filter for Label 4 to prove they exist
label_4_items = final_train_dataset.filter(lambda x: x["label"] == 4)
print(f"Proof: Found {len(label_4_items)} items with Label 4")

print(f"Final dataset size: {len(final_train_dataset)} rows")

# Save the merged and shuffled dataset back to a file
final_train_dataset.to_json("./dataset/wvsu_dataset_merged.jsonl")

print("Successfully saved merged data to ./dataset/wvsu_dataset_merged.jsonl")
