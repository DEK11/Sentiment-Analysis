from gsu.Sentiment import Sentiment
from gsu.Authentication import Authentication
from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import json
import os


class Listener(StreamListener):

    def __init__(self):
        super().__init__()
        self.s = Sentiment()
        try:
            os.remove("saved/twitter-out.txt")
        except OSError:
            pass

    def on_data(self, data):
        all_data = json.loads(data)
        # print("\n\nstarts\n\n", all_data)
        if all_data["lang"] != "en":
            return True
        tweet = all_data["text"]
        # print("\n\nneeded\n\n", tweet, "\n\nEnded\n\n")
        sentiment_value, confidence = self.s.Analyse(tweet)
        print(tweet, sentiment_value, confidence)

        if confidence*100 >= 80:
            output = open("saved/twitter-out.txt", "a")
            output.write(sentiment_value)
            output.write('\n')
            output.close()
        return True

    def on_error(self, status):
        print(status)


def main():
    cred = Authentication()
    auth = OAuthHandler(cred.ckey, cred.csecret)
    auth.set_access_token(cred.atoken, cred.asecret)
    twitterStream = Stream(auth, Listener())
    twitterStream.filter(track=["Obama"])


if __name__ == '__main__':
    main()
