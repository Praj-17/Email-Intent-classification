import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from loguru import logger
import sys
import os # Added for directory creation

# --- Updated Loguru Configuration ---
# Ensure 'logs' directory exists
logs_dir = "logs"
if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)
    print(f"Created directory: {logs_dir}") # Simple print for this setup step

logger.remove() # Remove default handler
logger.add(sys.stderr, format="{time} {level} {message}", level="INFO")
logger.add(os.path.join(logs_dir, "training.log"), rotation="500 MB", level="INFO") # Log to logs/training.log
# --- End Updated Loguru Configuration ---

logger.info("Starting training script...")

# 1. Load dataset
logger.info("Step 1: Loading dataset...")
DF = pd.read_csv("data/data.csv")  # columns: Email Number,Intent Category,...,Email Body
DF = DF.rename(columns={"Intent Category": "label", "Email Body": "text"})
DF['label'] = DF['label'].astype(str)
logger.info(f"Dataset loaded. Shape: {DF.shape}")

# 2. Create label map
logger.info("Step 2: Creating label map...")
labels = sorted(DF['label'].unique())
label2id = {lbl: idx for idx, lbl in enumerate(labels)}
id2label = {idx: lbl for lbl, idx in label2id.items()}
with open("label_map.json", "w") as f:
    json.dump(label2id, f, indent=2)
logger.info(f"Label map created and saved to label_map.json. Number of labels: {len(labels)}")

# 3. Encode labels
logger.info("Step 3: Encoding labels...")
DF['label_id'] = DF['label'].map(label2id)
logger.info("Labels encoded.")

def load_dataset(df):
    # returns HF Dataset-like dict
    return {
        'text': df['text'].tolist(),
        'label': df['label_id'].tolist()
    }

# Stratified split
logger.info("Performing stratified split of the dataset...")
train_df, val_df = train_test_split(
    DF, test_size=0.2, stratify=DF['label_id'], random_state=42
)
train_data = load_dataset(train_df)
val_data = load_dataset(val_df)
logger.info(f"Dataset split. Training size: {len(train_df)}, Validation size: {len(val_df)}")

# 4. Tokenizer & Model init
logger.info("Step 4: Initializing Tokenizer & Model...")
MODEL_NAME = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(labels)
)
logger.info(f"Tokenizer and Model ({MODEL_NAME}) initialized.")

# 5. Tokenization helper
logger.info("Step 5: Defining tokenization helper and tokenizing datasets...")
def tokenize_batch(batch):
    return tokenizer(
        batch['text'],
        padding='max_length',
        truncation=True,
        max_length=128
    )

# Convert to Datasets
from datasets import Dataset
train_dataset = Dataset.from_dict(train_data).map(tokenize_batch, batched=True)
val_dataset = Dataset.from_dict(val_data).map(tokenize_batch, batched=True)
train_dataset = train_dataset.remove_columns(['text'])
val_dataset = val_dataset.remove_columns(['text'])
train_dataset.set_format('torch', columns=['input_ids','attention_mask','label'])
val_dataset.set_format('torch', columns=['input_ids','attention_mask','label'])
logger.info("Datasets tokenized and formatted.")

# 6. TrainingArguments & Trainer
logger.info("Step 6: Setting up TrainingArguments & Trainer...")

# Define batch size here to use for steps_per_epoch calculation
# This value was previously hardcoded inside TrainingArguments
actual_per_device_train_batch_size = 8 
# Calculate steps per epoch
if len(train_df) == 0 or actual_per_device_train_batch_size == 0:
    steps_per_epoch = 1 # Avoid division by zero, set a default
    logger.warning("Training dataframe is empty or batch size is zero. Defaulting steps_per_epoch to 1.")
else:
    steps_per_epoch = len(train_df) // actual_per_device_train_batch_size

logger.info(f"Using eval_steps and save_steps for compatibility with older Transformers versions.")
logger.info(f"Calculated steps_per_epoch for eval/save: {steps_per_epoch} (based on train_df size {len(train_df)} and batch size {actual_per_device_train_batch_size})")

training_args = TrainingArguments(
    output_dir='./outputs',
    eval_strategy="steps", # Corrected from evaluation_strategy to eval_strategy
    # save_strategy default is "steps", which matches eval_strategy="steps"
    eval_steps=steps_per_epoch,    # Evaluate at the end of each epoch
    save_steps=steps_per_epoch,    # Save at the end of each epoch
    learning_rate=2e-5,
    per_device_train_batch_size=actual_per_device_train_batch_size,
    per_device_eval_batch_size=8,  # This can also be a variable if preferred
    num_train_epochs=5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model='f1',
    save_total_limit=2,
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='macro')
    return {'accuracy': acc, 'f1': f1}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)
logger.info("TrainingArguments and Trainer configured.")

# 7. Train & Save
logger.info("Step 7: Starting model training...")
trainer.train()
logger.info("Model training completed.")
logger.info("Saving model and tokenizer...")
trainer.save_model('./app/model')
tokenizer.save_pretrained('./app/model')
logger.info("Model and tokenizer saved to ./app/model.")

# 8. Evaluation Report
logger.info("Step 8: Evaluating model and generating report...")
preds_output = trainer.predict(val_dataset)
preds = preds_output.predictions.argmax(axis=1)
labels_true = preds_output.label_ids
report = classification_report(labels_true, preds, target_names=labels, output_dict=True)
logger.info(f"Classification Report:\n{classification_report(labels_true, preds, target_names=labels)}")
logger.info(f"Confusion Matrix:\n{confusion_matrix(labels_true, preds)}")
logger.info("Evaluation completed.")
logger.info("Training script finished.")