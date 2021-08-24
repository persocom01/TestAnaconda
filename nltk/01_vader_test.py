# Vader is a rule based sentiment analysis tool that is specifically attuned to
# sentiments expressed in social media.
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer

import_path = r'./datasets/reddit.csv'
data = pd.read_csv(import_path, index_col=None)
df = data[['title', 'subreddit']]


def sentiment_scores(sentence):
    sia = SentimentIntensityAnalyzer()
    sentiment_dict = sia.polarity_scores(sentence)
    return sentiment_dict


def classify_sentiment(sentiment_dict, threshold=0.05):
    if sentiment_dict['compound'] >= threshold:
        return 'pos'
    elif sentiment_dict['compound'] <= -threshold:
        return 'neg'
    else:
        return 'neu'


sentence = df['title'][0]
print('sentence: ' + sentence)
sentiment_dict = sentiment_scores(sentence)
print('sentiment dict: ' + str(sentiment_dict))
print('overall sentiment: ' + classify_sentiment(sentiment_dict))
