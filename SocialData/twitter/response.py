import config
from requests_oauthlib import OAuth1Session
import datetime, time

CK = config.CONSUMER_KEY
CS = config.CONSUMER_SECRET
AT = config.ACCESS_TOKEN
ATS = config.ACCESS_TOKEN_SECRET
twitter = OAuth1Session(CK, CS, AT, ATS) #認証処理

url = "https://api.twitter.com/1.1/statuses/user_timeline.json" #タイムライン取得エンドポイント

res = twitter.get(url)

limit = res.headers['x-rate-limit-remaining'] #リクエスト可能残数の取得
reset = res.headers['x-rate-limit-reset'] #リクエスト叶残数リセットまでの時間(UTC)
sec = int(res.headers['X-Rate-Limit-Reset']) - time.mktime(datetime.datetime.now().timetuple()) #UTCを秒数に変換

print ("limit: " + limit)
print ("reset: " +  reset)
print ('reset sec:  %s' % sec)
# https://qiita.com/bakira/items/00743d10ec42993f85eb