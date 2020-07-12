import urllib
from requests_oauthlib import OAuth1
import requests
import sys
import config

def main():

    CK = config.CONSUMER_KEY
    CKS = config.CONSUMER_SECRET
    AT = config.ACCESS_TOKEN
    ATS = config.ACCESS_TOKEN_SECRET

    # 検索時のパラメーター
    word = '別れました ありがとう' # 検索ワード
    count = 100 # 一回あたりの検索数(最大100/デフォルトは15)
    Range = 5 # 検索回数の上限値(最大180/15分でリセット)
   # ツイート検索・テキストの抽出
    tweets = search_tweets(CK, CKS, AT, ATS, word, count, Range)
    # 検索結果を表示
    print(tweets[0:4])


def search_tweets(CK, CKS, AT, ATS, word, count, Range):
    # 文字列設定
    word += ' exclude:retweets' # RTは除く
    word = urllib.parse.quote_plus(word)
    # リクエスト
    url = "https://api.twitter.com/1.1/search/tweets.json?lang=ja&q="+word+"&count="+str(count)
    auth = OAuth1(CK, CKS, AT, ATS)
    response = requests.get(url, auth=auth)
    data = response.json()['statuses']
    # 2回目以降のリクエスト
    cnt = 0
    tweets = []
    while True:
        if len(data) == 0:
            break
        cnt += 1
        if cnt > Range:
            break
        for tweet in data:
            tweets.append(tweet['text'])
            maxid = int(tweet["id"]) - 1
        url = "https://api.twitter.com/1.1/search/tweets.json?lang=ja&q="+word+"&count="+str(count)+"&max_id="+str(maxid)
        response = requests.get(url, auth=auth)
        try:
            data = response.json()['statuses']
        except KeyError: # リクエスト回数が上限に達した場合のデータのエラー処理
            print('上限まで検索しました')
            break
    return tweets


if __name__ == '__main__':
    main()
# https://qiita.com/h_tashiro/items/ed119c237f5595c3d7b8