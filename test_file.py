from transformers import pipeline
import tweepy

auth = tweepy.OAuth1UserHandler("v6VqqAplvd36dvUc4JRqB8Ejv", "1ZTfklxy01yfV7RiQznGiboFXmJqgf4rsGRhOVNGiNwXhrU17T")
auth.set_access_token("1763085504269139968-vvsuytPxI4Xy2Et82bM3K7VXG5D0lv", "khOLkZXt88fh9g5S79gDFLy4UAci3lzZ1o2NAALunsbSe")

#set up the pipeline
sentiment_analysis = pipeline("sentiment_analysis")

#define the stream listener
class MyStreamListiner(tweepy.StreamListener):
    def on_status(self, status):
        tweet_text = status.text


        sentiment = sentiment_pipeline(tweet_text[:512])
        print(f"Tweet: {tweet_text} \ nSentiment: {sentiment}")

my_listener = MyStreamListiner()
my_stream = tweepy.Stream(auth = api.auth, listener = my_listener)

#start the stream
my_stream.filter(track = ['keyword1', 'keyword2'])