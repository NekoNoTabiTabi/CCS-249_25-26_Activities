# Fine-tuning a Pretrained LLM for Emotion Classification

# Step 1: Identified the task to be performed.
# Task: Text Classification - Classify text into emotional categories (sadness, joy, love, anger, fear, surprise)

# Step 2: Identified the domain used for fine-tuning.
# Domain: Emotional text analysis - texts expressing various human emotions

# Step 3: Identified the LLM to be used.
# LLM: DistilBERT-base-uncased (a lightweight version of BERT for faster training)

# Step 4: Established the configuration needed for fine-tuning.
# - Model: distilbert-base-uncased
# - Dataset: emotion dataset from Hugging Face
# - Training: 3 epochs, batch size 16, learning rate 2e-5
# - Evaluation: Accuracy and weighted F1-score

# Step 5: Performed Evaluation, depending on the task performed.
# Evaluation metrics: Accuracy and weighted F1-score on validation and test sets

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)
import evaluate
import numpy as np

# Load the emotion dataset
dataset = load_dataset("emotion")

# The dataset has train, validation, and test splits
print("Dataset splits:", dataset.keys())
print("Train size:", len(dataset["train"]))
print("Validation size:", len(dataset["validation"]))
print("Test size:", len(dataset["test"]))

# Get label mappings
labels = dataset["train"].features["label"].names
id2label = {i: label for i, label in enumerate(labels)}
label2id = {label: i for i, label in enumerate(labels)}
num_labels = len(labels)

print("Labels:", labels)
print("Number of labels:", num_labels)

# Load tokenizer and model
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id
)

# Tokenize function
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)

# Tokenize datasets
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Load metrics
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"],
        "f1_weighted": f1.compute(predictions=preds, references=labels, average="weighted")["f1"],
    }

# Training arguments
training_args = TrainingArguments(
    output_dir="./emotion-distilbert-output",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    logging_steps=10,
    report_to="none"
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Train
trainer.train()

# Evaluate on validation
val_metrics = trainer.evaluate(eval_dataset=tokenized_datasets["validation"])
print("Validation Metrics:", val_metrics)

# Evaluate on test
test_metrics = trainer.evaluate(eval_dataset=tokenized_datasets["test"])
print("Test Metrics:", test_metrics)

# Save model
trainer.save_model("./emotion-distilbert-final")
tokenizer.save_pretrained("./emotion-distilbert-final")

# Quick inference demo
sample_texts = [
    "I am so happy to see you!",
    "This is the worst day ever.",
    "I feel scared about the future.",
    "I love this new book.",
    "I'm furious about this situation."
]

inputs = tokenizer(sample_texts, truncation=True, padding=True, return_tensors="pt")
outputs = model(**inputs)
preds = outputs.logits.argmax(dim=-1).tolist()

print("\nInference Results:")
for text, pred in zip(sample_texts, preds):
    print("TEXT: {}".format(text))
    print("PREDICTED EMOTION: {}".format(id2label[pred]))