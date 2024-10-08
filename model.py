import streamlit as st
from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
import numpy as np
from sklearn.metrics import accuracy_score
import os

# Function to compute model accuracy
def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}

def train_model():
    # Load IMDB dataset
    print("Loading IMDB dataset...")
    dataset = load_dataset('imdb')

    print("Dataset loaded.")

    # Select a portion of the data (e.g., 100 samples)
    train_dataset = dataset['train'].shuffle(seed=42).select(range(100))
    test_dataset = dataset['test'].shuffle(seed=42).select(range(100))

    # Load BERT tokenizer
    print("Loading BERT tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Tokenize the data to fit BERT input format
    print("Tokenizing the dataset...")
    def tokenize_function(example):
        return tokenizer(example['text'], padding="max_length", truncation=True, max_length=512)

    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_test = test_dataset.map(tokenize_function, batched=True)

    # Load BERT model for sequence classification
    print("Loading BERT model...")
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

    # Training settings
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=5,
        logging_dir="./logs",
        logging_steps=10,
    )

    # Trainer setup
    print("Setting up the trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        compute_metrics=compute_metrics,
    )

    # Train the model
    print("Starting model training...")
    trainer.train()

    # Save the model and tokenizer
    print("Saving the model...")

    save_directory = os.path.join(os.getcwd(), 'model-test')
    os.makedirs(save_directory, exist_ok=True)

    model.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)
    
    return trainer.evaluate()

if __name__ == "__main__":
    accuracy = train_model()
    print(f"Model accuracy: {accuracy['eval_accuracy']}")
