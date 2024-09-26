import pandas as pd
import numpy as np
import re
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
from sklearn.feature_extraction.text import TfidfVectorizer

# Load dataset
news_data = pd.read_csv('C:/Users/SAMHITHA/bigdata/train.csv')

# Data preprocessing
def clean_text(text):
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\d', ' ', text)
    text = text.lower()
    return text

news_data['cleaned_text'] = news_data['text'].apply(lambda x: clean_text(str(x)))
X = news_data['cleaned_text']
y = news_data['label']

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=5000)
X_tfidf = tfidf.fit_transform(X).toarray()

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Create DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, (hn, cn) = self.lstm(x)
        out = self.fc(out[:, -1, :])
        out = self.sigmoid(out)
        return out

# Initialize and train the model
model = LSTMModel(input_dim=5000, hidden_dim=128, output_dim=1)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(5):
    model.train()
    for inputs, labels in train_loader:
        outputs = model(inputs.unsqueeze(1))
        loss = criterion(outputs.squeeze(), labels.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# Evaluation
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for inputs, labels in test_loader:
        outputs = model(inputs.unsqueeze(1))
        predicted = (outputs.squeeze() > 0.5).long()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f'Test Accuracy: {accuracy * 100:.2f}%')

# Function to clean and preprocess the new input
def clean_and_tfidf_transform(text, tfidf):
    # Clean the text
    cleaned_text = clean_text(text)
    
    # Transform using the trained TF-IDF vectorizer
    text_tfidf = tfidf.transform([cleaned_text]).toarray()
    
    # Convert to PyTorch tensor
    text_tensor = torch.tensor(text_tfidf, dtype=torch.float32)
    
    return text_tensor

# Function to make predictions
def predict_news_lstm(model, tfidf, news):
    # Preprocess and transform the input news
    news_tensor = clean_and_tfidf_transform(news, tfidf)
    
    # Set the model to evaluation mode
    model.eval()
    
    # Make prediction
    with torch.no_grad():
        output = model(news_tensor.unsqueeze(1))
        prediction = (output.squeeze() > 0.5).long().item()
    
    # Return the prediction result
    return "Fake News" if prediction == 1 else "Real News"

# Example usage:
sample_news = "The government has approved new guidelines for COVID-19 vaccines."
result = predict_news_lstm(model, tfidf, sample_news)
print("Prediction:", result)

