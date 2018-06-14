import tweepy
import time
import json
import csv

# Twitter API Cred
consumer_key = "jFe6KYGfUDEwqC3KsrDjwsOfa"
consumer_secret = "m90xmY7M9BFIqZu29CzJzERY7vfSvkSJrLsV4bT3SvWqRrol7R"
access_token = "1001944086637629440-uDQvh8KprL4srOzajjxUWaMjBHPtsr"
access_secret = "gEW0u8DB3oQ8HPQQtmyKExl05y5Yfbd1yXugRxvwxCxTj"

# Establish connection using OAuth
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)
api = tweepy.API(auth)

# get Data from the profile
def get_profile(screen_name):
    api = tweepy.API(auth)
    try:
        user_profile = api.get_user(screen_name)
    except tweepy.error.TweepError as e:
        user_profile = json.loads(e.response.text)

    return  user_profile

def get_trends(location_id):
    api = tweepy.API(auth)
    try:
        trends = api.trends_place(location_id)
    except tweepy.error.TweepError as e:
        trends = json.loads(e.response.text)

    return trends

def get_tweets(query):
    api = tweepy.API(auth)
    try:
        tweets = api.search(query, count=100)
    except tweepy.error.TweepError as e:
        tweets = [json.loads(e.response.text)]

    return tweets

tw = get_tweets("#HanSolo")

# Saving to csv
queries = ["#Infinitywar"]

with open ("tweets.csv", "w", encoding="utf-8") as outfile:
    writer = csv.writer(outfile)
    writer.writerow(["id", "user", "created_at", "text"])
    for query in queries:
        t = get_tweets(query)
        for tweet in t:
            writer.writerow([tweet.id_str, tweet.user.screen_name,
                             tweet.created_at, tweet.text])