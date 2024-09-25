from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
import numpy as np
from sklearn.metrics import accuracy_score

# Function to compute model accuracy
def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}

def train_model():
    # Load IMDB dataset
    dataset = load_dataset('imdb')

    # Load BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Process data to fit BERT
    def tokenize_function(example):
        return tokenizer(example['text'], padding="max_length", truncation=True, max_length=512)

    tokenized_train = dataset['train'].map(tokenize_function, batched=True)
    tokenized_test = dataset['test'].map(tokenize_function, batched=True)

    # Load BERT model for classification
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

    # Training settings
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=1,
        logging_dir="./logs",
        logging_steps=10,
        fp16=True,
    )

    # Set up the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        compute_metrics=compute_metrics,
    )

    # Train the model
    trainer.train()

    # Save the model and tokenizer
    model.save_pretrained("./model")
    tokenizer.save_pretrained("./model")

    return trainer.evaluate()

if __name__ == "_main_":
    accuracy = train_model()
    print(f"Model accuracy: {accuracy['eval_accuracy']}")