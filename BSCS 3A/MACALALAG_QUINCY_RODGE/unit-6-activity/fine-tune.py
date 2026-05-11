# This codebase was generated with the help of AI

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)
import evaluate
import numpy as np
import random

# Specify the pre-trained model to use as the base for fine-tuning
model_name = "distilbert-base-uncased"

# Class labels for the agricultural text classification task
id2label = {
    0: "pest",
    1: "disease",
    2: "soil",
    3: "irrigation",
}
label2id = {v: k for k, v in id2label.items()}

# Example dataset of agricultural texts with labels for fine-tuning
examples = [
    {
        "text": "Rice field scouting found stem borer damage in tillering stage with dead hearts in scattered patches.",
        "label": 0,
    },
    {
        "text": "Maize leaves show chewing injury and frass near the whorl, suggesting fall armyworm infestation.",
        "label": 0,
    },
    {
        "text": "Aphids were clustered on the underside of eggplant leaves, causing curling and sticky honeydew deposits.",
        "label": 0,
    },
    {
        "text": "Brown planthopper populations increased after continuous flooding and dense rice planting.",
        "label": 0,
    },
    {
        "text": "Farm technicians observed leaf folder larvae folding rice leaves and feeding inside the folded blade.",
        "label": 0,
    },
    {
        "text": "Tomato fruits had puncture marks and larvae inside, indicating fruit borer attack.",
        "label": 0,
    },
    {
        "text": "The farmer reported whiteflies spreading quickly in the okra plot during hot dry weather.",
        "label": 0,
    },
    {
        "text": "Cutworms were severing newly transplanted cabbage seedlings at the base during the night.",
        "label": 0,
    },
    {
        "text": "Thrips damage caused silvery streaks on onion leaves and reduced vigor in young plants.",
        "label": 0,
    },
    {
        "text": "Weevils bored into stored corn grain, leaving powdery residues and exit holes.",
        "label": 0,
    },
    {
        "text": "Leaf miners created winding tunnels across the leaf surface of the bean crop.",
        "label": 0,
    },
    {
        "text": "Spider mites caused bronzing and fine webbing on drought-stressed peanut plants.",
        "label": 0,
    },
    {
        "text": "Rice leaves developed spindle-shaped lesions with gray centers, a classic symptom of blast disease.",
        "label": 1,
    },
    {
        "text": "Tomato plants showed yellowing, wilting, and dark vascular tissue consistent with bacterial wilt.",
        "label": 1,
    },
    {
        "text": "Banana leaves displayed yellow streaks and necrotic patches associated with sigatoka infection.",
        "label": 1,
    },
    {
        "text": "Powdery white fungal growth covered cucumber leaves during humid mornings.",
        "label": 1,
    },
    {
        "text": "Corn plants had elongated gray leaf spots that expanded rapidly after repeated rain.",
        "label": 1,
    },
    {
        "text": "Cassava stems and leaves exhibited mosaic patterns and distorted growth.",
        "label": 1,
    },
    {
        "text": "Pepper seedlings collapsed at the base due to damping-off in the nursery tray.",
        "label": 1,
    },
    {
        "text": "Soybean foliage showed rust pustules on the lower leaf surface and premature defoliation.",
        "label": 1,
    },
    {
        "text": "Citrus trees had canker-like raised lesions with yellow halos on leaves and fruit.",
        "label": 1,
    },
    {
        "text": "Papaya fruits developed water-soaked lesions that turned soft and sunken after harvest.",
        "label": 1,
    },
    {
        "text": "The potato plot showed late blight symptoms with dark lesions and white growth under moist conditions.",
        "label": 1,
    },
    {
        "text": "Peanut leaves presented circular brown spots that merged and caused heavy leaf drop.",
        "label": 1,
    },
    {
        "text": "Soil test results showed low nitrogen, low organic matter, and strongly acidic pH in the topsoil.",
        "label": 2,
    },
    {
        "text": "The field has compacted clay soil with poor drainage and restricted root penetration.",
        "label": 2,
    },
    {
        "text": "Laboratory analysis reported phosphorus deficiency and moderate potassium availability.",
        "label": 2,
    },
    {
        "text": "Salinity levels were elevated in the irrigated plot, causing poor seedling establishment.",
        "label": 2,
    },
    {
        "text": "The farm adviser recommended lime application because soil pH was below 5.0.",
        "label": 2,
    },
    {
        "text": "Soil texture in the vegetable bed was sandy loam with low water-holding capacity.",
        "label": 2,
    },
    {
        "text": "Nutrient mapping revealed zinc deficiency in several sections of the rice farm.",
        "label": 2,
    },
    {
        "text": "Root development was limited by a hardpan layer detected beneath the plow depth.",
        "label": 2,
    },
    {
        "text": "The compost-amended plot had higher organic carbon and better crumb structure.",
        "label": 2,
    },
    {
        "text": "Excessive soil acidity may be reducing nutrient uptake in the corn field.",
        "label": 2,
    },
    {
        "text": "The extension report noted boron deficiency symptoms linked to poor soil micronutrient balance.",
        "label": 2,
    },
    {
        "text": "Waterlogging persisted because the soil profile had slow infiltration and weak internal drainage.",
        "label": 2,
    },
    {
        "text": "Alternate wetting and drying was recommended for rice to reduce water use without lowering yield.",
        "label": 3,
    },
    {
        "text": "The irrigation schedule should be shifted to early morning to reduce evaporation losses.",
        "label": 3,
    },
    {
        "text": "Drip irrigation lines were installed to improve water efficiency in the tomato field.",
        "label": 3,
    },
    {
        "text": "Canal delivery was delayed, so the farmer adjusted transplanting to match water availability.",
        "label": 3,
    },
    {
        "text": "Moisture sensors indicated the root zone was still wet, so irrigation was postponed.",
        "label": 3,
    },
    {
        "text": "Furrow irrigation caused uneven water distribution across the sloping field.",
        "label": 3,
    },
    {
        "text": "The advisory suggested shorter but more frequent irrigation during flowering stage.",
        "label": 3,
    },
    {
        "text": "Pump discharge was insufficient to meet peak water demand in the dry season.",
        "label": 3,
    },
    {
        "text": "Mulching was combined with drip irrigation to conserve moisture in the vegetable plot.",
        "label": 3,
    },
    {
        "text": "Over-irrigation increased standing water and may have worsened root stress.",
        "label": 3,
    },
    {
        "text": "The reservoir level dropped sharply, requiring stricter irrigation rotation among farmers.",
        "label": 3,
    },
    {
        "text": "Sprinkler coverage was inconsistent because of low pressure and clogged nozzles.",
        "label": 3,
    },
]

# Splitting the dataset into training and validation sets, and preparing it for fine-tuning
random.seed(42)
random.shuffle(examples)

split_idx = int(len(examples) * 0.8)
train_data = examples[:split_idx]
valid_data = examples[split_idx:]

train_dataset = Dataset.from_list(train_data)
valid_dataset = Dataset.from_list(valid_data)

# Tokenizing the text data using the pre-trained model's tokenizer and preparing it for training

tokenizer = AutoTokenizer.from_pretrained(model_name)


def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True)


train_dataset = train_dataset.map(tokenize_function, batched=True)
valid_dataset = valid_dataset.map(tokenize_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=len(id2label), id2label=id2label, label2id=label2id
)

# Since this is a text classification, we use accuracy and F1 score as evaluation metrics during fine-tuning
# Instead of Perplexity, which is more common for language modeling tasks
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"],
        "f1_weighted": f1.compute(
            predictions=preds, references=labels, average="weighted"
        )["f1"],
    }


# Training configuration and execution of the fine-tuning process using the Hugging Face Trainer API
training_args = TrainingArguments(
    output_dir="./agri-distilbert-output",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=6,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    logging_steps=5,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

metrics = trainer.evaluate()
print(metrics)

trainer.save_model("./agri-distilbert-final")
tokenizer.save_pretrained("./agri-distilbert-final")

# quick inference demo
sample_docs = [
    "Rice leaves have diamond-shaped lesions and the disease is spreading after rain.",
    "Soil analysis shows low nitrogen and acidic pH across the field.",
    "Drip lines reduced water loss and improved irrigation efficiency in tomato rows.",
]

inputs = tokenizer(sample_docs, truncation=True, padding=True, return_tensors="pt")
outputs = model(**inputs)
preds = outputs.logits.argmax(dim=-1).tolist()

for text, pred in zip(sample_docs, preds):
    print(f"\nTEXT: {text}\nPREDICTED LABEL: {id2label[pred]}")
