# Import libraries
import tweepy
import numpy
import pandas
import matplotlib
from textblob import TextBlob
import re 
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator # Word clouds
from PIL import Image
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
# Import Twitter authentication file
from twitter_auth import *

# Twitter authentication
def twitter_auth():
    auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
    auth.set_access_token(ACCESS_TOKEN, ACCESS_SECRET)
    return tweepy.API(auth)

# Retrieve Tweets
def get_tweets():
    account = 'elonmusk' 
    extractor = twitter_auth()
    tweets = []
    for tweet in tweepy.Cursor(extractor.user_timeline, id = account).items():
        tweets.append(tweet._json['text'])
    print('Number of Tweets extracted: {}.\n'.format(len(tweets)))
    return tweets

# Create a dataframe containing the text in the Tweets
def make_dataframe(tweets):
    return pandas.DataFrame(data = tweets, columns = ['Tweets'])

# Pre-process Tweets
def text_preprocess(df):
    stop_list = stopwords.words('english')

    df['cleaned_tweets'] = df['Tweets'].str.lower()
    #Remove hyperlinks
    df['cleaned_tweets']= df['cleaned_tweets'].str.replace('https?://[A-Za-z0-9./]+', "")

    #Remove user mentions
    df['cleaned_tweets']= df['cleaned_tweets'].str.replace('@[A-Za-z0-9]+', "")

    #remove special characters/emojis
    df['cleaned_tweets']= df['cleaned_tweets'].str.replace('[^a-zA-Z\s]', "")
    
    nltk.download('stopwords')
    stop_list = stopwords.words('english')
    df['cleaned_tweets'] = df['cleaned_tweets'].apply(lambda x: " ".join([word for word in x.split() if word not in stop_list]))
    st = PorterStemmer()
    df['cleaned_tweets'] = df['cleaned_tweets'].apply(lambda x: " ".join([st.stem(word) for word in x.split()]))

    return df['cleaned_tweets']

# Retrieve sentiment of Tweet
def data_sentiment(tweet):
    # Returns 1, 0, or -1 depending on the value of text.sentiment.polarity
    text = TextBlob(str(tweet))
    if text.sentiment.polarity > 0:
        return 1
    elif text.sentiment.polarity == 0:
        return 0
    else:
        return -1

# Classify Tweets as good, neutral and bad tweets
def classify_tweets(data):
    good_tweets = ''
    neutral_tweets = ''
    bad_tweets = ''
    for index, row in data.iterrows():
        if row['sentiment'] > 0:
            good_tweets = good_tweets + row['cleaned_tweets']
        elif row['sentiment'] == 0:
            neutral_tweets = neutral_tweets + row['cleaned_tweets']
        else:
            bad_tweets = bad_tweets + row['cleaned_tweets']
    return [good_tweets, neutral_tweets, bad_tweets]

# Create word cloud for each type of tweets
def create_word_cloud(classified_tweets) :
    good_tweets = classified_tweets[0]
    neutral_tweets = classified_tweets[1]
    bad_tweets = classified_tweets[2]
    stopwords = set(STOPWORDS)
    good_cloud = WordCloud(good_tweets)
    neutral_cloud = WordCloud(neutral_tweets)
    bad_cloud = WordCloud(bad_tweets)

    good_cloud = WordCloud(width = 800, height = 500, background_color='green').generate(good_tweets)
    neutral_cloud = WordCloud(width = 800, height = 500, background_color='white').generate(neutral_tweets)
    bad_cloud = WordCloud(width = 800, height = 500, background_color='pink').generate(bad_tweets)
    
    produce_plot(good_cloud, "Good.png")
    produce_plot(neutral_cloud, "Neutral.png")
    produce_plot(bad_cloud, "Bad.png")

# Produce plot
def produce_plot(cloud, name):
    matplotlib.pyplot.axis("off")
    matplotlib.pyplot.imshow(cloud, interpolation='bilinear')
    fig = matplotlib.pyplot.figure(1)
    fig.savefig(name)
    matplotlib.pyplot.clf()

# Retrieve Tweets
tweets = get_tweets()

# Create dataframe 
df = make_dataframe(tweets)
df.head()

# Pre-process Tweets
df['cleaned_tweets'] = text_preprocess(df)
df.head()

# Retrieve sentiments

i=0
df['sentiment'] = 0
for tweet in df['cleaned_tweets']:
    df['sentiment'][i] = data_sentiment(tweet)
    i += 1

# Classify Tweets
classified_tweets = classify_tweets(df)

# Create Word Cloud
create_word_cloud(classified_tweets)

