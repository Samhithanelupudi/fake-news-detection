import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

# Load dataset (fake and real news)
news_data = pd.read_csv(r"C:\Users\SAMHITHA\bigdata\train.csv")

# Display the first few rows
print(news_data.head())

# Check for missing values
print(news_data.isnull().sum())

# Drop rows with missing data
news_data.dropna(inplace=True)

# Check the dataset info
print(news_data.info())

# Dataset contains columns: 'id', 'title', 'author', 'text', 'label'
# 'label' -> 1 (fake), 0 (real)


#data preprocessing
# Function to clean the text data
def clean_text(text):
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = re.sub(r'\d', ' ', text)   # Remove digits
    text = text.lower()  # Convert to lowercase
    return text

# Apply the cleaning function to the 'text' column
news_data['cleaned_text'] = news_data['text'].apply(lambda x: clean_text(str(x)))

# Drop the columns we don't need for training
news_data.drop(['id', 'author', 'title'], axis=1, inplace=True)

# Display a sample cleaned text
print(news_data['cleaned_text'].head())



#feature extraction
# Convert the text into numerical representation using TF-IDF
tfidf = TfidfVectorizer(stop_words='english', max_df=0.7)  # max_df ignores overly common words

# Prepare the target (label) and feature (text)
X = news_data['cleaned_text']  # The cleaned news text
y = news_data['label']  # The labels (0: real, 1: fake)

# Apply TF-IDF on the features
X_tfidf = tfidf.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)


#training the model
# Train the Logistic Regression model
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = lr_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Classification report
print("Classification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

#hyper parameter tuning with gridsearch CV
# Create a pipeline with TF-IDF and Logistic Regression
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_df=0.7)),
    ('clf', LogisticRegression(max_iter=1000))
])

# Define hyperparameters for tuning
param_grid = {
    'tfidf__max_df': [0.7, 0.8, 0.9],
    'clf__C': [0.1, 1, 10]  # Regularization strength
}

# Perform grid search
grid_search = GridSearchCV(pipeline, param_grid, cv=5, verbose=1, n_jobs=-1)
grid_search.fit(X, y)

# Best parameters and accuracy
print("Best Parameters:", grid_search.best_params_)
print("Best Cross-validation Accuracy:", grid_search.best_score_)


#making predictions
def predict_news(model, tfidf, news):
    # Clean and preprocess the new input
    cleaned_news = clean_text(news)
    
    # Convert the input into a format the model can understand
    transformed_news = tfidf.transform([cleaned_news])
    
    # Make a prediction
    prediction = model.predict(transformed_news)
    
    # Return prediction
    if prediction[0] == 1:
        return "Fake News"
    else:
        return "Real News"

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
#SENTIMENT ANALYSIS
# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Function to get sentiment score
def get_sentiment(text):
    score = analyzer.polarity_scores(text)
    return score['compound']

# Apply sentiment analysis
news_data['sentiment_score'] = news_data['text'].apply(lambda x: get_sentiment(str(x)))

# Display sentiment scores
print(news_data[['text', 'sentiment_score']].head())


# Test with a new piece of news
sample_news = "The government has approved new guidelines for COVID-19 vaccines."
result = predict_news(lr_model, tfidf, sample_news)
print("Prediction:", result)


