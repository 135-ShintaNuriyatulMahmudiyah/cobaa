import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load your dataset (replace 'your_dataset.csv' with your actual dataset)
df = pd.read_csv('your_dataset.csv')

# Streamlit app
st.title('Sentiment Analysis with Logistic Regression')

# Data Preprocessing
st.header('Data Preprocessing')
st.subheader('Original Dataset:')
st.write(df.head())

# Assuming you have a column named 'ulasan' containing the text data
# You can customize the preprocessing steps based on your dataset
# Example: Lowercasing and removing punctuation
df['clean_ulasan'] = df['ulasan'].str.lower().replace('[^\w\s]', '', regex=True)

# Display the preprocessed data
st.subheader('Preprocessed Dataset:')
st.write(df[['clean_ulasan', 'ulasan']].head())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['clean_ulasan'], df['sentiment'], test_size=0.2, random_state=42)

# Logistic Regression Model
st.header('Logistic Regression Model')

# Choose a vectorizer (CountVectorizer or TfidfVectorizer)
vectorizer_choice = st.radio('Choose a Vectorizer:', ('CountVectorizer', 'TfidfVectorizer'))

if vectorizer_choice == 'CountVectorizer':
    vectorizer = CountVectorizer()
elif vectorizer_choice == 'TfidfVectorizer':
    vectorizer = TfidfVectorizer()

# Build the pipeline with vectorizer and logistic regression model
model = make_pipeline(vectorizer, LogisticRegression())
model.fit(X_train, y_train)

# Model Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Display the evaluation metrics
st.header('Model Evaluation Metrics:')
st.write(f'Model Accuracy: {accuracy:.2%}')
st.text('Classification Report:')
st.text(classification_rep)
st.text('Confusion Matrix:')
st.write(conf_matrix)

# Sentiment Analysis for user input
st.header('Sentiment Analysis for User Input')
user_input = st.text_area('Enter a text for sentiment analysis:')
if st.button('Analyze Sentiment'):
    if user_input:
        user_input = pd.Series(user_input)
        sentiment_prediction = model.predict(user_input)
        st.subheader('Sentiment Prediction:')
        st.write(sentiment_prediction[0])
    else:
        st.warning('Please enter a text for sentiment analysis.')
