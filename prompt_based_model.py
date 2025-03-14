# Step 1: Install Required Libraries
!pip install transformers datasets torch scikit-learn nltk

# Step 2: Import Libraries
import os
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from nltk.translate.bleu_score import corpus_bleu
import zipfile
from google.colab import drive, files
from datasets import Dataset
import pandas as pd

os.environ["WANDB_DISABLED"] = "true"

# Step 3: Mount Google Drive
drive.mount('/content/drive')

# Step 4: Define Model Directories
MODEL_DIR_SCORING = "/content/drive/MyDrive/t5-scoring-final"
MODEL_DIR_FEEDBACK = "/content/drive/MyDrive/t5-feedback-final"

os.makedirs(MODEL_DIR_SCORING, exist_ok=True)
os.makedirs(MODEL_DIR_FEEDBACK, exist_ok=True)

# Step 5: Load Dataset
data = pd.read_csv("prompt.tsv", delimiter='\t')  # Replace with your file path

# Select relevant columns
data = data[['essay', 'content', 'organization', 'language', 'total',
             'content_feedback', 'organization_feedback', 'language_feedback']]

# Drop rows with missing values
data = data.dropna()

# Combine all feedback into a single column (optional)
data['combined_feedback'] = (
    data['content_feedback'] + " " +
    data['organization_feedback'] + " " +
    data['language_feedback']
)

# Combine all scores into a single column (optional)
data['combined_scores'] = (
    f"Content: {data['content']} "
    f"Organization: {data['organization']} "
    f"Language: {data['language']} "
    f"Total: {data['total']}"
)

# Step 6: Preprocess the Data for Scoring
def preprocess_scoring(examples):
    inputs = [f"Score this essay: {essay}" for essay in examples['essay']]
    targets = examples['combined_scores']
    return {'input_text': inputs, 'target_text': targets}

scoring_dataset = Dataset.from_pandas(data)
scoring_dataset = scoring_dataset.map(preprocess_scoring, batched=True)

# Step 7: Preprocess the Data for Feedback
def preprocess_feedback(examples):
    inputs = [f"Generate feedback for this essay: {essay}" for essay in examples['essay']]
    targets = examples['combined_feedback']
    return {'input_text': inputs, 'target_text': targets}

feedback_dataset = Dataset.from_pandas(data)
feedback_dataset = feedback_dataset.map(preprocess_feedback, batched=True)

# Step 8: Tokenize the Data
tokenizer = T5Tokenizer.from_pretrained("t5-small")

def tokenize_function(examples):
    inputs = tokenizer(examples['input_text'], max_length=512, truncation=True, padding="max_length")
    targets = tokenizer(examples['target_text'], max_length=128, truncation=True, padding="max_length")
    inputs["labels"] = targets["input_ids"]
    return inputs

tokenized_scoring = scoring_dataset.map(tokenize_function, batched=True)
tokenized_feedback = feedback_dataset.map(tokenize_function, batched=True)

# Step 9: Fine-Tune T5 for Scoring
scoring_model = T5ForConditionalGeneration.from_pretrained("t5-small").to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

training_args_scoring = TrainingArguments(
    output_dir=MODEL_DIR_SCORING,
    eval_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=1,  # Save only the final model
    logging_dir=f"{MODEL_DIR_SCORING}/logs",
    save_strategy="epoch",
    load_best_model_at_end=True,
    fp16=True,  # Enable mixed precision training
    report_to="none"  # Disable Weights & Biases logging
)

trainer_scoring = Trainer(
    model=scoring_model,
    args=training_args_scoring,
    train_dataset=tokenized_scoring,
    eval_dataset=tokenized_scoring,
    tokenizer=tokenizer,
    compute_metrics=None
)

trainer_scoring.train()

# Save the final scoring model
scoring_model.save_pretrained(MODEL_DIR_SCORING)
tokenizer.save_pretrained(MODEL_DIR_SCORING)

# Step 10: Fine-Tune T5 for Feedback
feedback_model = T5ForConditionalGeneration.from_pretrained("t5-small").to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

training_args_feedback = TrainingArguments(
    output_dir=MODEL_DIR_FEEDBACK,
    eval_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=1,  # Save only the final model
    logging_dir=f"{MODEL_DIR_FEEDBACK}/logs",
    save_strategy="epoch",
    load_best_model_at_end=True,
    fp16=True,  # Enable mixed precision training
    report_to="none"  # Disable Weights & Biases logging
)

trainer_feedback = Trainer(
    model=feedback_model,
    args=training_args_feedback,
    train_dataset=tokenized_feedback,
    eval_dataset=tokenized_feedback,
    tokenizer=tokenizer,
    compute_metrics=None
)

trainer_feedback.train()

# Save the final feedback model
feedback_model.save_pretrained(MODEL_DIR_FEEDBACK)
tokenizer.save_pretrained(MODEL_DIR_FEEDBACK)

# Step 11: Evaluate Scoring Model (Batched)
def evaluate_scoring(model, tokenizer, dataset, device, batch_size=16):
    predictions, references = [], []
    
    # Process in batches
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i + batch_size]
        inputs = [f"Score this essay: {example['essay']}" for example in batch]
        inputs_tokenized = tokenizer(inputs, return_tensors="pt", max_length=512, truncation=True, padding=True).to(device)
        
        # Generate predictions
        outputs = model.generate(**inputs_tokenized, max_length=64)
        decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        # Extract true scores
        true_scores = [example['combined_scores'] for example in batch]
        
        predictions.extend(decoded_outputs)
        references.extend(true_scores)
    
    # Convert scores to numerical values for evaluation
    pred_scores = [float(p.split(":")[-1].strip()) for p in predictions]
    true_scores = [float(r.split(":")[-1].strip()) for r in references]
    
    # Calculate metrics
    mae = mean_absolute_error(true_scores, pred_scores)
    rmse = mean_squared_error(true_scores, pred_scores, squared=False)
    r2 = r2_score(true_scores, pred_scores)
    
    print(f"Scoring Model Evaluation:\nMAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")

evaluate_scoring(scoring_model, tokenizer, tokenized_scoring, torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# Step 12: Evaluate Feedback Model (Batched)
def evaluate_feedback(model, tokenizer, dataset, device, batch_size=16):
    references, hypotheses = [], []
    
    # Process in batches
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i + batch_size]
        inputs = [f"Generate feedback for this essay: {example['essay']}" for example in batch]
        inputs_tokenized = tokenizer(inputs, return_tensors="pt", max_length=512, truncation=True, padding=True).to(device)
        
        # Generate predictions
        outputs = model.generate(**inputs_tokenized, max_length=128)
        decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        # Extract true feedback
        true_feedback = [example['combined_feedback'] for example in batch]
        
        hypotheses.extend(decoded_outputs)
        references.extend([[ref.split()] for ref in true_feedback])
    
    # Calculate BLEU score
    bleu = corpus_bleu(references, [hyp.split() for hyp in hypotheses])
    print(f"Feedback Model Evaluation:\nAverage BLEU Score: {bleu:.4f}")

evaluate_feedback(feedback_model, tokenizer, tokenized_feedback, torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# Step 13: Download Final Models
def zip_and_download(directory, zip_name):
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(directory):
            for file in files:
                zipf.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), directory))
    files.download(zip_name)

# Zip and download scoring model
zip_and_download(MODEL_DIR_SCORING, "t5-scoring-final.zip")

# Zip and download feedback model
zip_and_download(MODEL_DIR_FEEDBACK, "t5-feedback-final.zip")
