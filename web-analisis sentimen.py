import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load the preprocessed dataset
df = pd.read_csv('https://raw.githubusercontent.com/135-ShintaNuriyatulMahmudiyah/Data/main/data_baru.csv',sep='\t')
st.dataframe(df)')  # Replace with your preprocessed dataset

# Load the trained logistic regression model
logistic_regression_model = joblib.load('logistic_regression_model.joblib')  # Replace with your model path

# Load the chi-square feature selection model
chi2_model = joblib.load('chi2_feature_selection_model.joblib')  # Replace with your model path

# Streamlit app
st.title('Sentiment Analysis on Madura Tourism Data')

# Display some information about the dataset
st.subheader('Dataset Information:')
st.write(f"Number of Samples: {len(df)}")
st.write(f"Columns: {', '.join(df.columns)}")

# Feature selection using chi-square
X = df['ulasan']
y = df['sentiment']

# Vectorize the text data
vectorizer = TfidfVectorizer(max_features=5000)
X_vectorized = vectorizer.fit_transform(X)

# Perform chi-square feature selection
selected_features = chi2_model.transform(X_vectorized)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(selected_features, y, test_size=0.2, random_state=42)

# Train the logistic regression model
logistic_regression_model.fit(X_train, y_train)

# Evaluate the model
y_pred = logistic_regression_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

st.subheader('Model Performance Metrics:')
st.write(f'Model Accuracy: {accuracy:.2%}')
st.text('Classification Report:')
st.text(classification_rep)

# Sentiment analysis for user input
user_input = st.text_area('Enter your review:', '')
if st.button('Analyze Sentiment'):
    if user_input:
        # Preprocess user input
        user_input_vectorized = vectorizer.transform([user_input])
        user_input_selected_features = chi2_model.transform(user_input_vectorized)

        # Predict sentiment
        prediction = logistic_regression_model.predict(user_input_selected_features)[0]

        # Display the sentiment prediction
        st.subheader('Sentiment Prediction:')
        st.write(prediction)

    else:
        st.warning('Please enter a review for sentiment analysis.')
