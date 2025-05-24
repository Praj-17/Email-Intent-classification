import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, multilabel_confusion_matrix
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from loguru import logger
import sys
import os # Added for directory creation
import numpy as np # Added for multi-label compute_metrics
from sklearn.preprocessing import MultiLabelBinarizer # Added for multi-hot encoding

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

logger.info("Starting training script for multi-intent classification...")

# 1. Load dataset
logger.info("Step 1: Loading dataset...")
DF = pd.read_csv("data/data.csv")  # columns: Email Number,Intent Category,...,Email Body
# Rename 'Intent Category' to 'label_str' to handle original string labels
DF = DF.rename(columns={"Intent Category": "label_str", "Email Body": "text"})
# Convert label_str to list of strings. Assumes comma-separated if multiple, else wraps single string in list.
DF['label_list'] = DF['label_str'].apply(lambda x: [s.strip() for s in str(x).split(',')])
logger.info(f"Dataset loaded. Shape: {DF.shape}. Example label_list: {DF['label_list'].head().tolist()}")

# 2. Create label map and MultiLabelBinarizer
logger.info("Step 2: Creating label map and binarizer for multi-label...")
# Collect all unique labels from the lists of labels
all_labels_flat_list = sorted(list(set(label for sublist in DF['label_list'] for label in sublist)))

label2id = {label: i for i, label in enumerate(all_labels_flat_list)}
id2label = {i: label for i, label in enumerate(all_labels_flat_list)}
num_unique_labels = len(all_labels_flat_list)

# Initialize MultiLabelBinarizer with the sorted unique labels
mlb = MultiLabelBinarizer(classes=all_labels_flat_list)

with open("label_map.json", "w") as f:
    json.dump({"label2id": label2id, "id2label": id2label, "labels_ordered": all_labels_flat_list}, f, indent=2)
logger.info(f"Label map and binarizer created. Number of unique labels: {num_unique_labels}")
logger.info(f"Ordered labels for model: {all_labels_flat_list}")

# 3. Encode labels (Multi-hot encoding)
logger.info("Step 3: Encoding labels (multi-hot)...")
# Ensure labels are float32 for BCEWithLogitsLoss
DF['labels_encoded'] = [np.array(encoded_row, dtype=np.float32) for encoded_row in mlb.fit_transform(DF['label_list'])]
logger.info(f"Labels multi-hot encoded. Example: {DF['labels_encoded'].head().tolist()}")

def load_dataset_multilabel(df):
    # returns HF Dataset-like dict
    return {
        'text': df['text'].tolist(),
        'labels': df['labels_encoded'].tolist() # Ensure this is a list of multi-hot vectors
    }

# Stratified split for multi-label is complex.
# Using a simple split for now. For more robust splitting, consider scikit-multilearn's IterativeStratification.
# If you need to stratify, you might pick one label from each list (e.g., the first) as a proxy.
# DF['stratify_key'] = DF['label_list'].apply(lambda x: x[0] if x else 'N/A')
# For now, no stratification to keep it simple with multi-label.
logger.info("Performing train-test split of the dataset (no stratification for multi-label demo)...")
train_df, val_df = train_test_split(
    DF, test_size=0.2, random_state=42 # Consider stratify=DF['stratify_key'] if implemented
)
train_data = load_dataset_multilabel(train_df)
val_data = load_dataset_multilabel(val_df)
logger.info(f"Dataset split. Training size: {len(train_df)}, Validation size: {len(val_df)}")

# 4. Tokenizer & Model init
logger.info("Step 4: Initializing Tokenizer & Model for multi-label classification...")
MODEL_NAME = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_unique_labels,
    problem_type="multi_label_classification", # Crucial for multi-label
    label2id=label2id, # Pass mappings to model
    id2label=id2label
)
logger.info(f"Tokenizer and Model ({MODEL_NAME}) initialized for multi-label with {num_unique_labels} labels.")

# 5. Tokenization helper
logger.info("Step 5: Defining tokenization helper and tokenizing datasets...")
def tokenize_batch(batch):
    return tokenizer(
        batch['text'],
        padding='max_length',
        truncation=True,
        max_length=128 # Adjust if necessary
    )

# Convert to Datasets
from datasets import Dataset
train_dataset = Dataset.from_dict(train_data).map(tokenize_batch, batched=True)
val_dataset = Dataset.from_dict(val_data).map(tokenize_batch, batched=True)

train_dataset = train_dataset.remove_columns(['text'])
val_dataset = val_dataset.remove_columns(['text'])

# Set format: 'labels' should be torch.float for multi-label classification with BCEWithLogitsLoss
train_dataset.set_format('torch', columns=['input_ids','attention_mask','labels'], output_all_columns=True)
val_dataset.set_format('torch', columns=['input_ids','attention_mask','labels'], output_all_columns=True)
logger.info("Datasets tokenized and formatted with multi-hot labels as type float.")


# 6. TrainingArguments & Trainer
logger.info("Step 6: Setting up TrainingArguments & Trainer...")

actual_per_device_train_batch_size = 8
if len(train_df) == 0 or actual_per_device_train_batch_size == 0:
    steps_per_epoch = 1
    logger.warning("Training dataframe is empty or batch size is zero. Defaulting steps_per_epoch to 1.")
else:
    steps_per_epoch = len(train_df) // actual_per_device_train_batch_size

logger.info(f"Calculated steps_per_epoch for eval/save: {steps_per_epoch}")

training_args = TrainingArguments(
    output_dir='./outputs',
    eval_strategy="steps",
    eval_steps=steps_per_epoch,
    save_steps=steps_per_epoch,
    learning_rate=2e-5,
    per_device_train_batch_size=actual_per_device_train_batch_size,
    per_device_eval_batch_size=8,
    num_train_epochs=20, # Increased from 5 to 10
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model='f1_macro',
    save_total_limit=2,
    # greater_is_better=True by default for f1
)

def compute_metrics(eval_pred):
    logits, true_labels_multi_hot = eval_pred
    
    probs = 1 / (1 + np.exp(-logits)) # Sigmoid function
    
    # Log some raw probabilities for debugging (for a small subset of calls)
    # This will log during evaluations at each eval_step
    if hasattr(compute_metrics, "call_count"):
        compute_metrics.call_count += 1
    else:
        compute_metrics.call_count = 1

    if compute_metrics.call_count % 5 == 1: # Log roughly every 5th call to avoid excessive logging
        logger.info(f"Compute_metrics call #{compute_metrics.call_count}")
        logger.info(f"Debug eval: Sample true labels (first 2): {true_labels_multi_hot[:2].tolist()}")
        logger.info(f"Debug eval: Sample logits (first 2): {logits[:2].tolist()}")
        logger.info(f"Debug eval: Sample probabilities (first 2): {probs[:2].tolist()}")

    pred_labels_multi_hot = (probs > 0.5).astype(int)
    
    f1_macro = f1_score(true_labels_multi_hot, pred_labels_multi_hot, average='macro', zero_division=0)
    f1_micro = f1_score(true_labels_multi_hot, pred_labels_multi_hot, average='micro', zero_division=0)
    f1_samples = f1_score(true_labels_multi_hot, pred_labels_multi_hot, average='samples', zero_division=0)
    
    # Subset accuracy: (exact match of predicted label set with true label set)
    subset_accuracy = accuracy_score(true_labels_multi_hot, pred_labels_multi_hot)
    
    logger.debug(f"Sample true_labels_multi_hot: {true_labels_multi_hot[0].tolist() if len(true_labels_multi_hot)>0 else 'N/A'}")
    logger.debug(f"Sample pred_labels_multi_hot: {pred_labels_multi_hot[0].tolist() if len(pred_labels_multi_hot)>0 else 'N/A'}")


    return {
        'f1_macro': f1_macro,
        'f1_micro': f1_micro,
        'f1_samples': f1_samples,
        'subset_accuracy': subset_accuracy
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)
logger.info("TrainingArguments and Trainer configured for multi-label.")

# 7. Train & Save
logger.info("Step 7: Starting model training...")
trainer.train()
logger.info("Model training completed.")
logger.info("Saving model and tokenizer...")
trainer.save_model('./src/model')
tokenizer.save_pretrained('./src/model')
logger.info("Model and tokenizer saved to ./src/model.")

# 8. Evaluation Report
logger.info("Step 8: Evaluating model and generating report...")
preds_output = trainer.predict(val_dataset)
logits = preds_output.predictions
true_labels_multi_hot = preds_output.label_ids

# Apply sigmoid and threshold to get final predictions from logits
probs = 1 / (1 + np.exp(-logits))
pred_labels_multi_hot = (probs > 0.5).astype(int)

# Log final evaluation probabilities more extensively for the first few samples
logger.info(f"Final Eval: Sample true labels (first 3): {true_labels_multi_hot[:3].tolist()}")
logger.info(f"Final Eval: Sample logits (first 3): {logits[:3].tolist()}")
logger.info(f"Final Eval: Sample probabilities (first 3): {probs[:3].tolist()}")
logger.info(f"Final Eval: Sample predicted_multi_hot (first 3): {pred_labels_multi_hot[:3].tolist()}")


# Use all_labels_flat_list (ordered list of string labels) for target_names
report_str = classification_report(true_labels_multi_hot, pred_labels_multi_hot, target_names=all_labels_flat_list, zero_division=0)
logger.info(f"Classification Report (Multi-Label):\n{report_str}")

# Multi-label confusion matrix (one matrix per label)
mcm = multilabel_confusion_matrix(true_labels_multi_hot, pred_labels_multi_hot)
logger.info(f"Multi-label Confusion Matrices (one per label):")
for i, label_name in enumerate(all_labels_flat_list):
    logger.info(f"Confusion Matrix for label '{label_name}':\nTN: {mcm[i, 0, 0]} FP: {mcm[i, 0, 1]}\nFN: {mcm[i, 1, 0]} TP: {mcm[i, 1, 1]}")

logger.info("Evaluation completed.")
logger.info("Training script for multi-intent classification finished.")