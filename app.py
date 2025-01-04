from flask import Flask, request, render_template

import joblib
tfidf_1 = joblib.load('tfidf_vectorizer.pkl')
model = joblib.load('sentiment_model.pkl')

# Import necessary functions for analysis
from Sentiment_Analysis_Twitter import (
    fetch_tweets, preprocess_tweets, predict_sentiment, calculate_rating
)  # Replace 'your_analysis_module' with the file where these functions are defined

app = Flask(__name__)

# Home route to display the input form
@app.route('/')
def home():
    return render_template('index.html')  # Ensure 'index.html' exists in the templates folder

# Route to handle sentiment rating
@app.route('/rate', methods=['POST'])
def rate_account():
    username = request.form['username'].strip()  # Get the username from the form
    if not username:
        return "Please enter a valid username!"

    tweets = fetch_tweets(username)
    if not tweets:  # If no tweets are found, show a message
        return f"No tweets found for @{username} or the username is invalid."

    preprocessed = preprocess_tweets(tweets)
    sentiments = predict_sentiment(model,preprocessed,  tfidf_1)
    rating = calculate_rating(sentiments)
    return f"The sentiment rating for @{username} is {rating}/3!"

if __name__ == '__main__':
    app.run(port=9020, debug=True)