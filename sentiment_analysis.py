import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report

# Preprocess text data
def preprocess_data(data):
    data['Text Tweet'] = data['Text Tweet'].str.lower()
    data['Text Tweet'] = data['Text Tweet'].str.replace(r'[^a-zA-Z\s]', '', regex=True)
    data['Text Tweet'] = data['Text Tweet'].str.replace('\d', '', regex=True)
    data['Text Tweet'] = data['Text Tweet'].str.replace('[^\w\s]', '', regex=True)
    data['Text Tweet'] = data['Text Tweet'].str.replace('_', '')
    data['Text Tweet'] = data['Text Tweet'].str.replace('usermention', '')
    data['Text Tweet'] = data['Text Tweet'].str.replace('providername', '')
    data['Text Tweet'] = data['Text Tweet'].str.replace('productname', '')
    data['Text Tweet'] = data['Text Tweet'].str.replace('url', '')
    return data

# Train models
def train_sentiment_models(data):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(data['Text Tweet'])
    y = data['Sentiment']
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Naive Bayes model
    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train)
    
    # Train SVM model
    svm_model = SVC(kernel='linear')
    svm_model.fit(X_train, y_train)
    
    # Return models, vectorizer, and test data for evaluation
    return nb_model, svm_model, vectorizer, X_test, y_test

# Predict sentiment
def predict_sentiment(model, vectorizer, text):
    text = text.lower()
    text_vector = vectorizer.transform([text])
    return model.predict(text_vector)[0]