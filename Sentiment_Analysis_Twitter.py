#!/usr/bin/env python
# coding: utf-8

# Continuation of first notebook to improve accuracy of the logistic regression model

# In[1]:


import pandas as pd
import numpy as np
import re
import emoji
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import nltk
import joblib


# In[2]:


df = pd.read_csv('data/Tweets.csv', encoding='latin1')


# In[3]:


df.drop_duplicates(inplace=True)
df = df.set_axis(['target','id','date','flag','user','text'], axis='columns')
df['text'] = df['text'].str.lower()


# In[4]:


# Define a function to remove mentions
def remove_mentions(text):
    return re.sub(r'@\w+', '', text)

df['text'] = df['text'].apply(remove_mentions)

def clean_text_v2(text):
    text = re.sub(r"http\S+|www\.\S+", "", text)  # Remove URLs
    text = re.sub(r"\w+@\w+\.com", "", text)     # Remove emails
     # Normalize repeated punctuation (! and ?)
    text = re.sub(r"!{2,}", "!", text)  # Replace multiple exclamation marks with one
    text = re.sub(r"\?{2,}", "?",text)  # Replace multiple question marks with one
    text = re.sub(r"[.,;:\"'`]", "", text)     # Remove punctuation  but keep ! and ?
    text = re.sub(r"[@\$%^&*\(\)\\/\+-_=\[\]\{\}<>]", "", text)  # Remove special chars

    text = emoji.demojize(text, delimiters=(" ", " "))
    return text.strip()
df['text'] = df['text'].apply(clean_text_v2)


# In[5]:


#80/20 training split
x_train, x_test, y_train, y_test = train_test_split(df['text'], df['target'])


# In[6]:


tfidf_1 = TfidfVectorizer(max_features=20000, ngram_range = (1,2)).fit(x_train)
joblib.dump(tfidf_1, 'tfidf_vectorizer.pkl')


# In[7]:


x_train_vectorized = tfidf_1.transform(x_train)
x_test_vectorized = tfidf_1.transform(x_test)


# In[8]:


model = LogisticRegression(solver='saga',max_iter=1000 )
model.fit(x_train_vectorized, y_train)
predictions = model.predict(x_test_vectorized)

print('Confusion Matrix: \n', confusion_matrix(y_test, predictions))

print('\nAccuracy: \n', round(accuracy_score(y_test, predictions) * 100, 2), '%')

joblib.dump(model, 'sentiment_model.pkl')

# In[9]:


import tweepy

# Twitter API credentials
#consumer_key = 'aMJGEW763fGc3HG1mfcr64gQ3'
#consumer_secret = 'C7nx9QNaglEEZhP2ajWKi5sMPSFFjSqeACL3qoOSwFR9098HRy'
#access_token = '1332143772-hwnlmeDQwqTwY2iKHbyVJWWwCy1Ro3L94HMgDQf'
#access_token_secret = 'vaDWFxHG3KUpb5KxegCdKLmLmnVWaRrFKkuhaagvcG4BX'

# Replace with your Twitter API v2 credentials
bearer_token = "AAAAAAAAAAAAAAAAAAAAADagxwEAAAAAOpOHRRErATbr9r0csluV71ahaTg%3Drwt2914TwkmzqRIGJtwudsEUutJ7QkeRDZvhxoVNkGvOUSp9z0"

# Initialize the Tweepy Client
client = tweepy.Client(bearer_token=bearer_token)

# Authenticate
#client = tweepy.Client(
   # consumer_key=consumer_key,
   # consumer_secret=consumer_secret,
   # access_token=access_token,
    #access_token_secret=access_token_secret
#)

# Fetch tweets
def fetch_tweets(username, max_results=10):
    try:
        print(f"Fetching tweets for @{username}")
        user = client.get_user(username=username)
        
        if not user or not user.data:  # Check if user or user.data is None
            print(f"User @{username} not found.")
            return []

        user_id = user.data.id
        tweets = client.get_users_tweets(
            id=user_id,
            max_results=max_results,
            tweet_fields=["created_at", "lang", "text"]
        )
        
        if not tweets or not tweets.data:  # Check if tweets.data is None
            print(f"No tweets found for @{username}.")
            return []

        return [tweet.text for tweet in tweets.data]
    except tweepy.errors.TweepyException as e:
        print(f"Error fetching tweets for @{username}: {e}")
        return []
    except AttributeError as e:
        print(f"AttributeError: {e} - Likely caused by an invalid username.")
        return []


# In[10]:


def preprocess_tweets(tweets):
    return [clean_text_v2(tweet) for tweet in tweets]


# In[11]:


def predict_sentiment(model, tweets, tfidf_1):
    tweets_vectorized = tfidf_1.transform(tweets)
    predictions = model.predict(tweets_vectorized)
    return predictions


# In[12]:


def calculate_rating(sentiments):
    sentiment_map = {0: 1, 2: 2, 4: 3} # Map sentiment labels to scores
    scores = [sentiment_map[sent] for sent in sentiments]
    return round(sum(scores) / len(scores), 2)  # Average rating
# %%
