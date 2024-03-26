import os
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AdamW
from torch.utils.data import DataLoader
from datasets import load_dataset
import pandas as pd
import re

# Data cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text

# Dataset class
class FinancialNewsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(int(self.labels[idx]))
        return item

    def __len__(self):
        return len(self.labels)

# Training function
def train():
    dataset = load_dataset("zeroshot/twitter-financial-news-topic")
    df_train = pd.DataFrame(dataset["train"])
    df_test = pd.DataFrame(dataset["validation"])

    # Apply text cleaning
    df_train['text'] = df_train['text'].apply(clean_text)
    df_test['text'] = df_test['text'].apply(clean_text)

    # Tokenization and model preparation
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=20)

    # Encoding and dataset preparation
    train_encodings = tokenizer(df_train['text'].tolist(), truncation=True, padding=True, max_length=64)  # Reduced max_length
    test_encodings = tokenizer(df_test['text'].tolist(), truncation=True, padding=True, max_length=64)  # Reduced max_length
    train_dataset = FinancialNewsDataset(train_encodings, df_train['label'].tolist())
    test_dataset = FinancialNewsDataset(test_encodings, df_test['label'].tolist())

    # DataLoader and optimizer setup
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)  # Increased batch size
    optim = AdamW(model.parameters(), lr=5e-5)

    # Training loop
    epochs = 1  # Keep it to one for testing, can be increased as needed
    model.train()
    for epoch in range(epochs):
        for batch in train_loader:
            optim.zero_grad()
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optim.step()

    # Save the trained model
    model.save_pretrained("/opt/ml/model")

if __name__ == "__main__":
    train()
