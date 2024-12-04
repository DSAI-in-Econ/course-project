import pandas as pd
import urllib.parse
import html
import requests
import json
import time
import re
import os

if 'BAIDU_API_KEY' not in os.environ or 'BAIDU_SECRET_KEY' not in os.environ:
    print('Please set BAIDU_API_KEY and BAIDU_SECRET_KEY in your environment variables.')
    exit()

API_KEY = os.getenv('BAIDU_API_KEY')
SECRET_KEY = os.getenv('BAIDU_SECRET_KEY')

posts = pd.read_csv('data/posts.csv')
posts.head()

# html decoding

posts['text'] = posts['text'].fillna('').apply(html.unescape)
posts = posts[posts['text'].str.len() > 50]

def get_title(text: str):
    para = text.split('\n')[0]
    if para.startswith('发布了头条文章'):
        # extract title between '《' and '》'
        start = para.find('《')
        end = para.find('》')
        if start != -1 and end != -1:
            return para[start+1:end]
    # if the first sentence is not longer than 80 bytes, return it
    para_len = len(bytes(para, encoding='utf-8'))
    if para_len <= 80:
        return para
    
    sentence = para.split('。')[0]
    if len(bytes(sentence, encoding='utf-8')) <= 80:
        return sentence
    else:
        return str(bytes(sentence, encoding='utf-8')[:80], encoding='utf-8', errors='ignore') + '...'
        
posts['title'] =  posts['text'].apply(get_title)
# refresh the index
posts = posts.reset_index(drop=True)

def get_access_token(key, secret):
    url = "https://aip.baidubce.com/oauth/2.0/token"
    params = {"grant_type": "client_credentials", "client_id": key, "client_secret": secret}
    return str(requests.post(url, params=params).json().get("access_token"))

def parse_topic(title, text, token):
        
    url = "https://aip.baidubce.com/rpc/2.0/nlp/v1/topic?charset=UTF-8&access_token=" + token
    
    payload = json.dumps({
        "content": text,
        "title": title
    })
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }
    
    response = requests.request("POST", url, headers=headers, data=payload)
    if response.status_code == 200 and 'item' in response.json():
        return response.json()['item']
    else:
        print(f'Error {response.status_code}: {response.text}')
        print(f'Title: {title}')
        print(f'Text: {text}')
        return {}
    
token = get_access_token(API_KEY, SECRET_KEY)

DELAY = 0.5

total = len(posts)
topics = []

for i, row in posts.iterrows():
    print(f'Processing {i} / {total}...')
    topics.append(parse_topic(row['title'], row['text'], token))
    time.sleep(DELAY)

posts['topic'] = [topic['lv1_tag_list'][0]['tag'] for topic in topics]
posts['topic_score'] = [topic['lv1_tag_list'][0]['score'] for topic in topics]

posts['subtopic_1'] = [topic['lv2_tag_list'][0]['tag'] if len(topic['lv2_tag_list']) > 0 else pd.NA for topic in topics]
posts['subtopic_1_score'] = [topic['lv2_tag_list'][0]['score'] if len(topic['lv2_tag_list']) > 0 else pd.NA for topic in topics]

posts['subtopic_2'] = [topic['lv2_tag_list'][1]['tag'] if len(topic['lv2_tag_list']) > 1 else pd.NA for topic in topics]
posts['subtopic_2_score'] = [topic['lv2_tag_list'][1]['score'] if len(topic['lv2_tag_list']) > 1 else pd.NA for topic in topics]

posts.to_csv('processed/posts.csv', index=False)

# replace <img alt="[鲜花]" title="[鲜花]" src="https://face.t.sinajs.cn/t4/appstyle/expression/ext/normal/d4/2018new_xianhua_org.png" /> with [鲜花]
comments = pd.read_csv('data/comments.csv')

comments['text'] = comments['text'].fillna('').apply(html.unescape)

def replace_emoji(text: str):
    return re.sub(r'<img alt="\[(.*?)\]" title="\[(.*?)\]" src=".*?" />', r'[\1]', text)

# replace <a href=/n/比尔盖茨 usercard="name=@比尔盖茨">@比尔盖茨</a> with @比尔盖茨
def replace_user(text: str):
    return re.sub(r'<a href=/n/(.*?) usercard="name=@(.*?)">@.*?</a>', r'@\2', text)

# replace <a href="//s.weibo.com/weibo?q=%23%E6%84%9F%E6%81%A9%E8%8A%82%23" target="_blank">#感恩节#</a> with #感恩节#  
def replace_topic(text: str):
    return re.sub(r'<a href="//s.weibo.com/weibo\?q=%23(.*?)%23" target="_blank">#.*?#</a>', r'#\1#', text)

# replace  <a target="_blank" href="https://weibo.com/p/100808bf477e2e55fee42ffffab602a712046b"><img class="icon-link" title="#英语学习[超话]#" src="https://h5.sinaimg.cn/upload/100/959/2020/05/09/timeline_card_small_super_default.png"/>英语学习超话</a> with #英语学习超话#
def replace_topic2(text: str):
    return re.sub(r'<a target="_blank" href="https://weibo.com/p/(.*?)"><img class="icon-link" title="#(.*?)#" src=".*?"/>\2</a>', r'#\2#', text)

# remove <span class="expand">全文</span>
def remove_expand(text: str):
    return text.replace('<span class="expand">全文</span>', '')

def preprocess_comment(text: str):
    text = html.unescape(text)
    text = replace_emoji(text)
    text = replace_user(text)
    text = remove_expand(text)
    text = replace_topic(text)
    text = replace_topic2(text)
    return text

comments['text'] = comments['text'].fillna('').apply(preprocess_comment)
comments['text'] = comments['text'].fillna('').apply(lambda x: urllib.parse.unquote(x))

# remove comments whois post_id is not in the posts
comments = comments[comments['post_id'].isin(posts['id'])]
# refresh the index
comments = comments.reset_index(drop=True) 

def get_comment_sentiment(text: str, token):
    url = "https://aip.baidubce.com/rpc/2.0/nlp/v1/sentiment_classify?charset=UTF-8&access_token=" + token
    
    payload = json.dumps({
        "text": text
    })
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }
    
    response = requests.request("POST", url, headers=headers, data=payload)
    # if 200 and is json response
    if response.status_code != 200:
        print(f'Error {response.status_code}: {response.text}')
        print(f'Text: {text}')
        return {}
    try :
        resp = response.json()['items'][0]
    except:
        print(f'Error not json {response.status_code}: {response.text}')
        print(f'Text: {text}')
        return {}
    
    if resp['sentiment'] == 0:
        sentiment = 'negative'
    elif resp['sentiment'] == 1:
        sentiment = 'neutral'
    else:
        sentiment = 'positive'
    return {
        'sentiment': sentiment,
        'confidence': resp['confidence']
    }

token = get_access_token(API_KEY, SECRET_KEY)

total = len(comments)

sentiments = []

DELAY = 0.5

for i, row in comments.iterrows():
    print(f'Processing {i} / {total}...')
    sentiments.append(get_comment_sentiment(row['text'], token))
    time.sleep(DELAY)

comments['sentiment'] = [sentiment['sentiment'] if sentiment else pd.NA for sentiment in sentiments]
comments['confidence'] = [sentiment['confidence'] if sentiment else pd.NA for sentiment in sentiments]

comments.to_csv('processed/comments.csv', index=False)


