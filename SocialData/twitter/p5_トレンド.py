import json, config #標準のjsonモジュールとconfig.pyの読み込み
from requests_oauthlib import OAuth1Session #OAuthのライブラリの読み込み
import twitter

CONSUMER_KEY = config.CONSUMER_KEY
CONSUMER_SECRET = config.CONSUMER_SECRET
OAUTH_TOKEN = config.ACCESS_TOKEN
OAUTH_TOKEN_SECRET = config.ACCESS_TOKEN_SECRET
auth = twitter.oauth.OAuth(OAUTH_TOKEN, OAUTH_TOKEN_SECRET,
                           CONSUMER_KEY, CONSUMER_SECRET)

twitter_api = twitter.Twitter(auth=auth)

WORLD_WOE_ID = 1
US_WOE_ID = 23424856
world_trends = twitter_api.trends.place(_id=WORLD_WOE_ID)
jp_trends = twitter_api.trends.place(_id=US_WOE_ID)

print(world_trends)
for i in jp_trends[0]['trends']:
    print(i['name'])