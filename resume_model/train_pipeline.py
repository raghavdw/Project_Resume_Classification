# train_pipeline.py

import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict, load_metric
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding
)
from resume_model.config.core import config
from resume_model.pipeline import resume_pipe
from resume_model.processing.data_manager import load_dataset, save_pipeline



def load_and_split_data(data_path):
    
    """Loads the dataset and performs the train-test-validation split."""
    df = load_dataset(file_name=config.app_config.training_data_file)

    train_test_df, val_df = train_test_split(df, test_size=0.1, random_state=42, stratify=df['label'])
    train_df, test_df = train_test_split(train_test_df, test_size=0.2, random_state=42, stratify=train_test_df['label'])

    return train_df, val_df, test_df

# 2. Dataset Preparation
def prepare_datasets(train_df, val_df, test_df):
    """Converts DataFrames to Hugging Face Datasets and creates a DatasetDict."""
    train_ds = Dataset.from_pandas(train_df)
    val_ds = Dataset.from_pandas(val_df)
    test_ds = Dataset.from_pandas(test_df)

    ds = DatasetDict()
    ds['train'] = train_ds
    ds['validation'] = val_ds
    ds['test'] = test_ds

    return ds

# 3. Tokenization
def tokenize_data(ds, checkpoint="distilbert-base-uncased"):
    """Tokenizes the 'Cleaned_Resume' column using the specified checkpoint."""
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    def tokenize_fn(batch):
        return tokenizer(batch['Cleaned_Resume'], truncation=True)

    tokenized_datasets = ds.map(tokenize_fn, batched=True)
    return tokenized_datasets

# 4. Model Loading and Configuration
def load_model(checkpoint="distilbert-base-uncased", num_labels=25):
    """Loads the pre-trained model and sets the number of labels."""
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=num_labels)
    return model

# 5. Metrics
def compute_metrics(logits_and_labels):
    """Computes the F1 score for evaluation."""
    from sklearn.metrics import f1_score

    logits, labels = logits_and_labels
    predictions = np.argmax(logits, axis=-1)
    return {'f1_score': f1_score(y_true=labels, y_pred=predictions, average='weighted')}

# 6. Training Setup and Execution
def train_model(model, tokenized_datasets, model_output_path="/content/bert_model"):
    """Sets up training arguments, data collator, and Trainer, then starts training."""
    training_args = TrainingArguments(
        output_dir=model_output_path,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        num_train_epochs=40  # Adjust as needed
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    trainer.train()

# Main Execution (if running this script directly)
if __name__ == "__main__":
    data_path = "your_data.csv"  # Replace with your actual data path
    train_df, val_df, test_df = load_and_split_data(data_path)
    ds = prepare_datasets(train_df, val_df, test_df)
    tokenized_datasets = tokenize_data(ds)
    model = load_model()
    train_model(model, tokenized_datasets)