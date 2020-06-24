#This still does not work because an account for developers from Twitter has not been created yet



import trynltk.sentiment_mod as sm

print(sm.sentiment("This movie was awesome, the acting was great, plot was wonderful."))
print(sm.sentiment("This movie was uter junk. I dont see what the point was at all. Horrible movie.Everything was stupid"))


from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import json

#consumer key, consumer secret, access token, access secret.
ckey=""
csecret=""
atoken=""
asecret=""

class listener(StreamListener):

    def on_data(self, data):
        all_data = json.loads(data)

        tweet = all_data["text"]

        print((tweet))

        return True

    def on_error(self, status):
        print (status)

auth = OAuthHandler(ckey, csecret)
auth.set_access_token(atoken, asecret)

twitterStream = Stream(auth, listener())
twitterStream.filter(track=["car"])