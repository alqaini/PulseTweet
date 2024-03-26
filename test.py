import re
import pandas as pd
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.utils.data import DataLoader
from datasets import load_dataset
import torch

# Data cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text

# Evaluation function
def evaluate(model, data_loader):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']
            outputs = model(input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=1)
            total += labels.size(0)
            correct += (predictions == labels).sum().item()
    return correct / total

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

# Main function for evaluation
def main():
    # Load the test dataset
    dataset = load_dataset("zeroshot/twitter-financial-news-topic", split='validation')
    df_test = pd.DataFrame(dataset)
    df_test['text'] = df_test['text'].apply(clean_text)

    # Load tokenizer and model
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained('./model')

    # Prepare dataset
    test_encodings = tokenizer(df_test['text'].tolist(), truncation=True, padding=True, max_length=64)
    test_dataset = FinancialNewsDataset(test_encodings, df_test['label'].tolist())
    test_loader = DataLoader(test_dataset, batch_size=8) 

    # Evaluate the model
    accuracy = evaluate(model, test_loader)
    print(f'Test Accuracy: {accuracy:.4f}')

if __name__ == "__main__":
    main()