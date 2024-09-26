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
3. Open the setting on your repository, choose 'Action' from the left panel menu then scroll down for 'Workflow permissions' to make sure that 'Read and write permissions' is choosen and check this option 'Allow GitHub Actions to create and approve pull requests' then click save. 
4. After creating the repository, clone it to your local machine:
   ```bash
   git clone https://github.com/<your-username>/<your-repo-name>/

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
   transformers
   datasets
   torch
   streamlit
   pytest
   scikit-learn
   transformers[torch]
   tf-keras

2. Create model.py<br><br>
This script will train the BERT model for sentiment analysis using the IMDB dataset. Below is the code for model.py:
   ``` bash
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
       # Load the IMDB dataset
       print("Loading IMDB dataset...")
       dataset = load_dataset('imdb')

       print("Dataset loaded.")

       # Select a portion of the data (e.g., 100 samples)
       train_dataset = dataset['train'].shuffle(seed=42).select(range(100))
       test_dataset = dataset['test'].shuffle(seed=42).select(range(100))

       # Load the BERT tokenizer
       print("Loading BERT tokenizer...")
       tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

       # Tokenize the data for BERT
       print("Tokenizing the dataset...")
       def tokenize_function(example):
           return tokenizer(example['text'], padding="max_length", truncation=True, max_length=512)

       tokenized_train = dataset['train'].map(tokenize_function, batched=True)
       tokenized_test = dataset['test'].map(tokenize_function, batched=True)

       # Load the BERT model for classification
       print("Loading BERT model...")
       model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

       # Training arguments
       training_args = TrainingArguments(
           output_dir="./results",
           evaluation_strategy="epoch",
           per_device_train_batch_size=8,
           per_device_eval_batch_size=8,
           num_train_epochs=5,
           logging_dir="./logs",
           logging_steps=10,
       )

       # Set up the Trainer
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

   if _name_ == "__main__":
       accuracy = train_model()
       print(f"Model accuracy: {accuracy['eval_accuracy']}")

3. Create app.py <br><br>
This script will deploy the Streamlit user interface, allowing users to analyze text sentiment using the trained model. Below is the code for app.py:<br><br>
   ``` bash
   import streamlit as st
   from transformers import BertTokenizer, BertForSequenceClassification
   import torch
   
   # Load the model and tokenizer
   model = BertForSequenceClassification.from_pretrained('./model-test')
   tokenizer = BertTokenizer.from_pretrained('./model-test')

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
       assert accuracy['eval_accuracy'] > 0.6, "Model accuracy is too low!"
   
## Handel Git Large File Storage (LFS)

### Step 4: Install Git LFS Locally
To handle large files in your GitHub repository and set up your CI/CD pipeline using GitHub Actions with Git Large File Storage (LFS) and BERT for your IMDB sentiment analysis project, follow these detailed steps:
1. Install Git LFS:
- Follow the instructions for your operating system from the [Git LFS installation page](https://git-lfs.github.com/).
2. Initialize Git LFS in Your Repository: Open your terminal or command prompt, navigate to your local repository, and run:
   ``` bash
   git lfs install
3. Track Large Files: Specify which file types you want Git LFS to manage. For your case, you want to track .safetensors files:
   ``` bash
   git lfs track "*.safetensors"
4. Commit the Changes: This will create a .gitattributes file in your repository. Commit this file:
   ``` bash
   git add .gitattributes
   git commit -m "Track large files with Git LFS"

## GitHub Actions Setup

### Step 5: Set Up GitHub Actions

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
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.x'  #change x to the compatible version of python woks with your code

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt

      - name: Train model
        run: python model.py
   
      - name: Add model files to git
        run: |
          git config --local user.email "your-github-email@example.com"
          git config --local user.name "your-github-username"
          git add model-test/
          git commit -m "Add trained model files" || echo "No changes to commit"
    
      - name: Push changes
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        run: |
          git push origin HEAD:main

      - name: Run tests
        run: pytest test_model.py
    
   #or you can use artifact to upload your model to the runner (workflow environment)
      #- name: Upload Model as Artifact
       # uses: actions/upload-artifact@v3
       # with:
         # name: model
          #path: ./model-test 


## Deploying with Streamlit Cloud

### Step 6: Deploying the Application Using Streamlit Cloud

1. Create an account on Streamlit Cloud.
2. Log in using your GitHub account.
3. Click on New app to create a new deployment.
4. Select your GitHub repository imdb-sentiment-analysis-with-bert.
5. Choose the main branch and set app.py as the main application file.
6. Click on Deploy.

**Streamlit Cloud will automatically install the required libraries from requirements.txt and run the application.**

## Final Result

### Step 7: Accessing the Deployed Application

Once deployed, you will receive a link to your application. You can use this link to access the application and input text for sentiment analysis using the trained BERT model.

## Conclusion

You now have a sentiment analysis application powered by a BERT model trained on the IMDB dataset. The application is set up for deployment using Streamlit Cloud, and all steps are automated with GitHub Actions for continuous integration and delivery.
