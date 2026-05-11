# FINE-TUNING A PRETRAINED LLM: Spam vs. Not-Spam Text Classification

# STEP 1: Task Identified    → Binary Text Classification (Spam Detection)
# STEP 2: Domain Identified  → Email / SMS Spam Filtering
# STEP 3: Model Identified   → distilbert-base-uncased (~66M params, fast)
# STEP 4: Configuration      → HuggingFace Trainer with SequenceClassification
# STEP 5: Evaluation         → Accuracy, Precision, Recall, F1, Confusion Matrix

# DATASET : ucirvine/sms_spam  (HuggingFace Hub)
# MODEL   : distilbert-base-uncased



import os
import json
import warnings
from dataclasses import dataclass

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    set_seed,
)
from datasets import load_dataset
from datasets.exceptions import DatasetNotFoundError
import evaluate
from sklearn.decomposition import PCA
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

warnings.filterwarnings("ignore")


#  STEP 1 - TASK DEFINITION
# Binary Text Classification — Spam vs not spam


TASK_NAME  = "Binary Text Classification"
LABEL2ID   = {"ham": 0, "spam": 1}
ID2LABEL   = {0: "ham", 1: "spam"}

print("=" * 70)
print(f"  STEP 1 — TASK   : {TASK_NAME}")
print(f"           LABELS : {list(LABEL2ID.keys())}  →  {list(LABEL2ID.values())}")
print("=" * 70)


#  STEP 2 - DOMAIN & DATASET
# DOMAIN : Email / SMS Spam Filtering

DOMAIN     = "Email / SMS Spam Filtering"
DATASET_ID = "ucirvine/sms_spam"

print(f"\n  STEP 2 — DOMAIN  : {DOMAIN}")
print(f"           DATASET : {DATASET_ID}")


def load_spam_dataset(dataset_id: str):
    print(f"\n  Loading '{dataset_id}'...")
    try:
        raw = load_dataset(dataset_id, split="train")
    except DatasetNotFoundError:
      
        if dataset_id == "ucirvine/sms_spam_collection":
            fallback_id = "ucirvine/sms_spam"
            print(f"  Dataset '{dataset_id}' not found. Falling back to '{fallback_id}'.")
            raw = load_dataset(fallback_id, split="train")
        else:
            raise

    
    col_map = {}
    if "sms" not in raw.column_names and "text" in raw.column_names:
        col_map["text"] = "sms"
    if col_map:
        raw = raw.rename_columns(col_map)

    if raw.features["label"].dtype == "string" or isinstance(
        raw[0]["label"], str
    ):
        raw = raw.map(lambda x: {"label": LABEL2ID[x["label"]]})

    split = raw.train_test_split(test_size=0.15, seed=42, stratify_by_column="label")
    val_test = split["test"].train_test_split(test_size=0.5, seed=42)

    print(f"  Train : {len(split['train'])} samples")
    print(f"  Val   : {len(val_test['train'])} samples")
    print(f"  Test  : {len(val_test['test'])} samples")

    spam_count = sum(1 for x in split["train"] if x["label"] == 1)
    ham_count  = len(split["train"]) - spam_count
    print(f"  Train balance — ham: {ham_count}, spam: {spam_count}")
    print(f"\n  Sample spam : {split['train'].filter(lambda x: x['label']==1)[0]['sms'][:80]}...")
    print(f"  Sample ham  : {split['train'].filter(lambda x: x['label']==0)[0]['sms'][:80]}...")

    return split["train"], val_test["train"], val_test["test"]


#  STEP 3 - MODEL SELECTION
# MODEL : distilbert-base-uncased  (HuggingFace Hub)

MODEL_ID = "distilbert-base-uncased"

print(f"\n  STEP 3 — MODEL  : {MODEL_ID}")
print(f"           PARAMS : ~66M  |  Arch: DistilBERT (encoder-only, bidirectional)")


#  STEP 4 - CONFIGURATION & FINE-TUNING

@dataclass
class SpamConfig:
    model_id: str           = MODEL_ID
    dataset_id: str         = DATASET_ID
    output_dir: str         = "./spam_classifier"
    max_length: int         = 128
    num_train_epochs: int   = 5
    per_device_train_batch_size: int = 32
    per_device_eval_batch_size: int  = 64
    learning_rate: float    = 2e-5
    weight_decay: float     = 0.01
    warmup_ratio: float     = 0.06
    fp16: bool              = torch.cuda.is_available()
    seed: int               = 42
    logging_steps: int      = 20
    eval_steps: int         = 50
    save_steps: int         = 50
    load_best_model_at_end: bool  = True
    metric_for_best_model: str    = "f1"   # F1 is key for imbalanced labels
    early_stopping_patience: int  = 2

CONFIG = SpamConfig()
set_seed(CONFIG.seed)
print(f"\n  STEP 4 — CONFIGURATION")
print(f"  {'─'*60}")
for k, v in CONFIG.__dict__.items():
    print(f"  {k:<35} : {v}")
print(f"  {'─'*60}")
print(f"  Device : {'CUDA (' + torch.cuda.get_device_name(0) + ')' if torch.cuda.is_available() else 'CPU'}")


# TOKENIZED DATASET 

class SpamDataset(Dataset):
    

    def __init__(self, hf_dataset, tokenizer, max_length: int = 128):
        self.labels = []
        self.encodings = tokenizer(
            [item["sms"] for item in hf_dataset],
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        self.labels = torch.tensor([item["label"] for item in hf_dataset])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels":         self.labels[idx],
        }


# METRICS FUNCTION

accuracy_metric = evaluate.load("accuracy")
f1_metric       = evaluate.load("f1")
precision_metric = evaluate.load("precision")
recall_metric    = evaluate.load("recall")


def compute_metrics(eval_pred):
    
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    acc = accuracy_metric.compute(predictions=predictions, references=labels)
    f1  = f1_metric.compute(predictions=predictions, references=labels,
                             average="binary", pos_label=1)
    prec = precision_metric.compute(predictions=predictions, references=labels,
                                    average="binary", pos_label=1)
    rec  = recall_metric.compute(predictions=predictions, references=labels,
                                  average="binary", pos_label=1)
    return {
        "accuracy":  acc["accuracy"],
        "f1":        f1["f1"],
        "precision": prec["precision"],
        "recall":    rec["recall"],
    }


# TRAINING PIPELINE 

def run_finetuning(cfg: SpamConfig):
    

     
    print("\n  [1/5] Loading dataset...")
    train_data, val_data, test_data = load_spam_dataset(cfg.dataset_id)

  
    print("\n  [2/5] Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_id,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params    : {total:,}")
    print(f"  Trainable params: {trainable:,}")


    print("\n  [3/5] Tokenizing datasets...")
    train_ds = SpamDataset(train_data, tokenizer, cfg.max_length)
    val_ds   = SpamDataset(val_data,   tokenizer, cfg.max_length)
    test_ds  = SpamDataset(test_data,  tokenizer, cfg.max_length)

     
    training_args = TrainingArguments(
        output_dir                  = cfg.output_dir,
        num_train_epochs            = cfg.num_train_epochs,
        per_device_train_batch_size = cfg.per_device_train_batch_size,
        per_device_eval_batch_size  = cfg.per_device_eval_batch_size,
        learning_rate               = cfg.learning_rate,
        weight_decay                = cfg.weight_decay,
        warmup_ratio                = cfg.warmup_ratio,
        fp16                        = cfg.fp16,
        eval_strategy               = "steps",
        eval_steps                  = cfg.eval_steps,
        save_strategy               = "steps",
        save_steps                  = cfg.save_steps,
        logging_steps               = cfg.logging_steps,
        load_best_model_at_end      = cfg.load_best_model_at_end,
        metric_for_best_model       = cfg.metric_for_best_model,
        greater_is_better           = True,   # higher F1 is better
        seed                        = cfg.seed,
        report_to                   = "none",
    )

    
    print("\n  [4/5] Training...")
    trainer = Trainer(
        model           = model,
        args            = training_args,
        train_dataset   = train_ds,
        eval_dataset    = val_ds,
        processing_class = tokenizer,
        compute_metrics = compute_metrics,
        callbacks       = [EarlyStoppingCallback(
                              early_stopping_patience=cfg.early_stopping_patience
                          )],
    )

    train_result = trainer.train()
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)

    print(f"\n  Model saved  → {cfg.output_dir}")
    print(f"  Train time   : {train_result.metrics.get('train_runtime', 0):.1f}s")

    return trainer, model, tokenizer, test_ds, test_data



#  STEP 5 - EVALUATION




def run_evaluation(trainer, model, tokenizer, test_ds, test_data, cfg: SpamConfig):
    
    print("\n  STEP 5 - EVALUATION ON TEST SET")
    print(f"  {'─'*60}")

   
    preds_output = trainer.predict(test_ds)
    logits       = preds_output.predictions
    labels       = preds_output.label_ids
    predictions  = np.argmax(logits, axis=-1)

   
    acc  = accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"]
    f1   = f1_metric.compute(predictions=predictions, references=labels,
                              average="binary", pos_label=1)["f1"]
    prec = precision_metric.compute(predictions=predictions, references=labels,
                                     average="binary", pos_label=1)["precision"]
    rec  = recall_metric.compute(predictions=predictions, references=labels,
                                  average="binary", pos_label=1)["recall"]

    print(f"\n  {'Metric':<20} {'Score':>8}")
    print(f"  {'─'*30}")
    print(f"  {'Accuracy':<20} {acc:>8.4f}")
    print(f"  {'Precision (spam)':<20} {prec:>8.4f}")
    print(f"  {'Recall (spam)':<20} {rec:>8.4f}")
    print(f"  {'F1 (spam)':<20} {f1:>8.4f}")

    
    print("\n  Full Classification Report:")
    print(classification_report(labels, predictions, target_names=["ham", "spam"]))

    
    cm = confusion_matrix(labels, predictions)
    tn, fp, fn, tp = cm.ravel()
    print(f"\n  Confusion Matrix:")
    print(f"  {'':>12} Pred Ham   Pred Spam")
    print(f"  {'Actual Ham':<12} {tn:>8}   {fp:>8}")
    print(f"  {'Actual Spam':<12} {fn:>8}   {tp:>8}")
    print(f"\n  TP={tp}  TN={tn}  FP={fp}  FN={fn}")
    print(f"  False Positive Rate : {fp/(fp+tn):.4f}  (legit mail sent to spam)")
    print(f"  False Negative Rate : {fn/(fn+tp):.4f}  (spam that slipped through)")

  
    print("\n Sample Predictions ")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    model.to(device)

    demo_texts = [
        "WINNER!! You have been selected for a $500 prize. Call now to claim!",
        "Hey, are we still on for lunch tomorrow at 1pm?",
        "FREE entry in 2 a weekly competition to win FA Cup final tkts!",
        "Can you pick up some milk on your way home?",
        "Urgent! Your account has been compromised. Click here immediately.",
        "Thanks for the birthday wishes! Had a great time last night.",
    ]

    for text in demo_texts:
        enc  = tokenizer(text, return_tensors="pt",
                         max_length=CONFIG.max_length,
                         truncation=True, padding="max_length").to(device)
        with torch.no_grad():
            logits = model(**enc).logits
        probs   = torch.softmax(logits, dim=-1)[0]
        pred_id = torch.argmax(probs).item()
        label   = ID2LABEL[pred_id]
        conf    = probs[pred_id].item()
        flag    = "🚫 SPAM" if pred_id == 1 else "✅ HAM "
        print(f"  {flag}  ({conf:.2%})  \"{text[:60]}...\"" if len(text) > 60
              else f"  {flag}  ({conf:.2%})  \"{text}\"")



    results = {
        "accuracy":  round(acc,  4),
        "precision": round(prec, 4),
        "recall":    round(rec,  4),
        "f1":        round(f1,   4),
        "confusion_matrix": {"TP": int(tp), "TN": int(tn),
                              "FP": int(fp), "FN": int(fn)},
    }
    os.makedirs(cfg.output_dir, exist_ok=True)
    report_path = os.path.join(cfg.output_dir, "evaluation_report.json")
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Report saved → {report_path}")
    print("\n" + "=" * 70)
    print("  EVALUATION COMPLETE")
    print("=" * 70)
    return results



def predict_spam(text: str, model_dir: str = CONFIG.output_dir) -> dict:
  
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model     = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    enc = tokenizer(text, return_tensors="pt", max_length=128,
                    truncation=True, padding="max_length")
    with torch.no_grad():
        logits = model(**enc).logits
    probs   = torch.softmax(logits, dim=-1)[0]
    pred_id = torch.argmax(probs).item()
    return {
        "label":      ID2LABEL[pred_id],
        "confidence": round(probs[pred_id].item(), 4),
        "scores":     {"ham": round(probs[0].item(), 4),
                       "spam": round(probs[1].item(), 4)},
    }

#  PCA VISUALIZATION 

def visualize_word_vectors_pca(model, tokenizer, words, output_dir: str):
    """Project word vectors to 2D with PCA and save a labeled scatter plot."""
    if len(words) < 20:
        raise ValueError("Provide at least 20 words for PCA visualization.")

    embed_weights = model.get_input_embeddings().weight.detach().cpu().numpy()
    vectors = []
    labels = []

    for word in words:
        token_ids = tokenizer(word, add_special_tokens=False).input_ids
        if not token_ids:
            continue
        vec = np.mean(embed_weights[token_ids], axis=0)
        vectors.append(vec)
        labels.append(word)

    if len(vectors) < 2:
        raise ValueError("Not enough tokenized words to plot PCA.")

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(np.vstack(vectors))

    os.makedirs(output_dir, exist_ok=True)
    fig_path = os.path.join(output_dir, "pca_word_vectors.png")

    plt.figure(figsize=(10, 7))
    plt.scatter(coords[:, 0], coords[:, 1], s=40, alpha=0.8)
    for i, label in enumerate(labels):
        plt.annotate(label, (coords[i, 0], coords[i, 1]),
                     textcoords="offset points", xytext=(4, 4), fontsize=9)

    plt.title("PCA of DistilBERT Input Embeddings")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=160)
    plt.show()

    print(f"\n  PCA plot saved → {fig_path}")



def main():
    print("\n" + "=" * 70)
    print("  SPAM CLASSIFIER — DistilBERT Fine-Tuning Pipeline")
    print("=" * 70)


    trainer, model, tokenizer, test_ds, test_data = run_finetuning(CONFIG)

 
    print("\n  [5/5] Evaluating on test set...")
    results = run_evaluation(trainer, model, tokenizer, test_ds, test_data, CONFIG)

   
    pca_words = [
        "money", "bank", "loan", "credit", "debit", "cash", "salary", "invoice",
        "market", "stock", "trade", "price", "profit", "loss", "risk", "budget",
        "tax", "bill", "payment", "transfer", "account", "balance", "invest", "refund",
    ]
    visualize_word_vectors_pca(model, tokenizer, pca_words, CONFIG.output_dir)

    return results


if __name__ == "__main__":
    main()
