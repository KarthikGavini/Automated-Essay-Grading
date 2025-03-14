# Step 0: Install Required Libraries
!pip install transformers datasets torch scikit-learn

# Step 1: Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Step 2: Import Libraries
import os
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import Dataset

# Disable tokenizer parallelism to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"


os.environ["WANDB_DISABLED"] = "true"

# Step 3: Load the Balanced Dataset
# Path to your dataset in Google Drive
dataset_path = "balanced_dataset.csv" 
df = pd.read_csv(dataset_path)

# Optional: Reduce dataset size for testing
# df = df.sample(n=500, random_state=42)

# Step 4: Tokenization
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize(texts, tokenizer, max_length=512):
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

# Step 5: Split Dataset into Train, Validation, and Test Sets
train_texts, temp_texts, train_labels, temp_labels = train_test_split(
    df["essay"].tolist(),
    df["score"].tolist(),
    test_size=0.2,
    random_state=42,
    stratify=df["score"]
)

val_texts, test_texts, val_labels, test_labels = train_test_split(
    temp_texts,
    temp_labels,
    test_size=0.5,
    random_state=42,
    stratify=temp_labels
)

# Step 6: Create PyTorch Dataset
class EssayDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

# Tokenize datasets
train_encodings = tokenize(train_texts, tokenizer)
val_encodings = tokenize(val_texts, tokenizer)
test_encodings = tokenize(test_texts, tokenizer)

train_dataset = EssayDataset(train_encodings, train_labels)
val_dataset = EssayDataset(val_encodings, val_labels)
test_dataset = EssayDataset(test_encodings, test_labels)

# Step 7: Load Pre-trained Model
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=6  # Six classes (scores 0–5)
)

# Move model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Step 8: Configure Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    fp16=True,
    run_name="essay_scoring_experiment",  # Unique run name
    report_to="none",  # Disable WandB
)

# Step 9: Define Metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }

# Step 10: Train the Model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# Start training
trainer.train()

# Step 11: Evaluate on Validation Set
val_results = trainer.evaluate()
print(f"Validation results: {val_results}")

# Step 12: Evaluate on Test Set
test_results = trainer.evaluate(test_dataset)
print(f"Test results: {test_results}")

# Step 13: Save the Final Model
# Save the model and tokenizer to Google Drive for persistence
MODEL_SAVE_PATH = "/content/drive/MyDrive/AES_Project/final_model"
trainer.save_model(MODEL_SAVE_PATH)
tokenizer.save_pretrained(MODEL_SAVE_PATH)
