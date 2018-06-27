import csv
import re
import pandas as pd
from bs4 import BeautifulSoup

# Clean the data
def clean_data(text):
    cleaned = []
    for t in text:
        t = (re.sub(r'@[A-Za-z0-9]+', '', t))       # remove @mentions
        t = re.sub('https?://[A-Za-z0-9./]+','',t)  # remove links
        t = re.sub("[^a-zA-Z]", " ", t)             # Remove numbers and punctuations
        t = BeautifulSoup(t, 'lxml')                # remove html encoded text
        t = t.text.replace("RT", "")
        t = t.lower()
        cleaned.append(t)
    return cleaned


def import_lexicons(path):
    # import lexicons
    lexicons = pd.read_csv(path, names=["words", "score"])
    lexicons = lexicons.drop_duplicates("words")
    return lexicons


# assign sentiments to tweets
def sentiment_analysis(cleaned, lexicons):
    sentiment = []
    sentiment_score = []
    for i in range(len(cleaned)):
        tweet_score = 0
        for w in cleaned[i].split():
            if w in list(lexicons.words):
                tweet_score = tweet_score + lexicons.score[lexicons.words == w].values
        sentiment_score.append(tweet_score)
        if tweet_score > 0:
            sentiment.append("positive")
        elif tweet_score < 0:
            sentiment.append("negative")
        else:
            sentiment.append("neutral")

    tweets_sentiments = pd.DataFrame(data={"text": tweets_data["text"],
                                           "sentiment": sentiment,
                                           "sentiment_score": sentiment_score})
    return tweets_sentiments


def write_to_csv(dataframe, path):
    dataframe.to_csv(path)


tweets_data = pd.read_csv("tweets.csv")
tweets_text = tweets_data["text"]
cleaned_data = clean_data(tweets_text)
lexicons = import_lexicons("lexicons.csv")
tweets_sentiments = sentiment_analysis(cleaned_data, lexicons)
write_to_csv(tweets_sentiments, "tweets_with_sentiments.csv")