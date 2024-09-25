# IMDB Sentiment Analysis with BERT

This project implements a complete sentiment analysis solution using the BERT (bert-base-uncased) model from Hugging Face's Transformers library. The sentiment analysis model is trained on the IMDB dataset, and a user-friendly interface is created using Streamlit for deployment. Additionally, GitHub Actions are used for continuous integration and delivery (CI/CD).

## Table of Contents
1. [Project Setup](#project-setup)
2. [Local Project Structure](#local-project-structure)
3. [Python Libraries and Model Setup](#python-libraries-and-model-setup)
4. [GitHub Actions Setup](#github-actions-setup)
5. [Deploying with Streamlit Cloud](#deploying-with-streamlit-cloud)
6. [Final Result](#final-result)

## Project Setup

### Step 1: Create a GitHub Project

1. Open [GitHub](https://github.com) and create a new repository.
2. Name the repository, for example, imdb-sentiment-analysis-with-bert.
3. After creating the repository, clone it to your local machine:
   ```bash
   git clone https://github.com/reemSameh21/imdb-sentiment-analysis-with-bert/

## Local Project Structure

### Step 2: Set Up the Project Locally

1. Inside your project directory, create the following files:
   - `requirements.txt`: This file will list the required libraries for the project.
   - `model.py`: This script will be used to train the sentiment analysis model using BERT.
   - `app.py`: This script will handle the user interface deployment with Streamlit.
   - `test_model.py`: This file will be used to test the model accuracy after training.

## Python Libraries and Model Setup

### Step 3: Set Up Python Libraries and the Model

1. Create requirements.txt<br><br>
   **Add the required libraries for the project:**<br>
   ``` bash
   <br>transformers
   <br>datasets
   <br>torch
   <br>streamlit
   <br>pytest
   <br>scikit-learn
   <br>transformers[torch]

2. Create model.py<br><br>
This script will train the BERT model for sentiment analysis using the IMDB dataset. Below is the code for model.py:
   ``` bash
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
       # Load the IMDB dataset
       dataset = load_dataset('imdb')

       # Load the BERT tokenizer
       tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

       # Tokenize the data for BERT
       def tokenize_function(example):
           return tokenizer(example['text'], padding="max_length", truncation=True, max_length=512)

       tokenized_train = dataset['train'].map(tokenize_function, batched=True)
       tokenized_test = dataset['test'].map(tokenize_function, batched=True)

       # Load the BERT model for classification
       model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

       # Training arguments
       training_args = TrainingArguments(
           output_dir="./results",
           evaluation_strategy="epoch",
           per_device_train_batch_size=8,
           per_device_eval_batch_size=8,
           num_train_epochs=3,  # Adjust as necessary
           logging_dir="./logs",
           logging_steps=10,
       )

       # Set up the Trainer
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

   if _name_ == "_main_":
       accuracy = train_model()
       print(f"Model accuracy: {accuracy['eval_accuracy']}")

3. Create app.py <br><br>
This script will deploy the Streamlit user interface, allowing users to analyze text sentiment using the trained model. Below is the code for app.py:<br><br>
   ``` bash
   import streamlit as st
   from transformers import BertTokenizer, BertForSequenceClassification
   import torch
   
   # Load the model and tokenizer
   model = BertForSequenceClassification.from_pretrained('./model')
   tokenizer = BertTokenizer.from_pretrained('./model')

   st.title("IMDB Sentiment Analysis with BERT")

   # User input for movie review
   user_input = st.text_area("Enter a movie review:")

   if st.button("Analyze"):
       if user_input:
           # Tokenize the user input
           inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True, max_length=512)
           with torch.no_grad():
               outputs = model(**inputs)
           prediction = torch.argmax(outputs.logits, dim=1).item()

           if prediction == 1:
               st.write("Sentiment: Positive")
           else:
               st.write("Sentiment: Negative")
   
4. Create test_model.py<br><br>
This file is for testing the model accuracy post-training:<br>
   ``` bash
   from model import train_model

   def test_model_accuracy():
       accuracy = train_model()
       assert accuracy['eval_accuracy'] > 0.7, "Model accuracy is too low!"
   
## GitHub Actions Setup

### Step 4: Set Up GitHub Actions

1. Create a directory named .github/workflows/ in the project root.
2. In this directory, create a file named ci-cd.yml and add the following configuration for CI/CD using GitHub Actions:
   ``` bash
   name: Model CI/CD with BERT and IMDB Dataset

    on:
      push:
        branches:
          - main

    jobs:
      build:
        runs-on: ubuntu-latest

      steps:
      - name: Checkout repository
        uses: actions/checkout@v2

      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.x'

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt

      - name: Train model
        run: python model.py

      - name: Run tests
        run: pytest test_model.py
   
## Deploying with Streamlit Cloud

### Step 5: Deploying the Application Using Streamlit Cloud

1. Create an account on Streamlit Cloud.
2. Log in using your GitHub account.
3. Click on New app to create a new deployment.
4. Select your GitHub repository imdb-sentiment-analysis-with-bert.
5. Choose the main branch and set app.py as the main application file.
6. Click on Deploy.

**Streamlit Cloud will automatically install the required libraries from requirements.txt and run the application.**

## Final Result

### Step 6: Accessing the Deployed Application

Once deployed, you will receive a link to your application. You can use this link to access the application and input text for sentiment analysis using the trained BERT model.

## Conclusion

You now have a sentiment analysis application powered by a BERT model trained on the IMDB dataset. The application is set up for deployment using Streamlit Cloud, and all steps are automated with GitHub Actions for continuous integration and delivery.
