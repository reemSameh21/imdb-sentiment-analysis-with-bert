import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load model and tokenizer
model = BertForSequenceClassification.from_pretrained('model')
tokenizer = BertTokenizer.from_pretrained('model')

st.title("IMDB Sentiment Analysis with BERT")

# User input
user_input = st.text_area("Enter a movie review:")

if st.button("Analyze"):
    if user_input:
        # Convert text to numerical representation using the tokenizer
        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()

        if prediction == 1:
            st.write("Sentiment: Positive")
        else:
            st.write("Sentiment: Negative")
