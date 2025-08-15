import streamlit as st
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from sklearn.preprocessing import LabelEncoder

# Load the trained PyTorch model and tokenizer
# Make sure the model and tokenizer are loaded in the previous cells
# For this example, we'll assume 'model', 'tokenizer', and 'label_encoder' are available
# You might need to save and load them properly in a real application

# Assuming the model, tokenizer, and label_encoder are available from previous cells
if 'model' not in globals() or 'tokenizer' not in globals() or 'label_encoder' not in globals():
    st.error("Model, tokenizer, or label encoder not found. Please run the previous cells.")
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval() # Set model to evaluation mode

    st.title("BBC News Article Classifier (PyTorch)")
    input_text = st.text_area("Enter your news article here:")

    if st.button('Classify'):
        if input_text:
            # Preprocess the input text
            encoded_input = tokenizer(input_text, truncation=True, padding=True, return_tensors='pt')
            input_ids = encoded_input['input_ids'].to(device)
            attention_mask = encoded_input['attention_mask'].to(device)

            # Get prediction from the model
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predicted_class_idx = torch.argmax(logits, dim=1).item()

            # Decode the predicted class
            predicted_category = label_encoder.inverse_transform([predicted_class_idx])[0]

            st.write(f"Predicted Category: {predicted_category}")
        else:
            st.warning("Please enter some text to classify.")