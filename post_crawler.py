import requests
import time
import json
import pandas as pd
import os

if 'COOKIE_SUB' not in os.environ:
    print('Please set the COOKIE_SUB environment variable')
    exit(1)

COOKIE_SUB = os.environ['COOKIE_SUB']
UID = 1743951792
DELAY = 0.5
TOTAL_PAGES = 1
USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
POSTS_ENDPOINT = 'https://weibo.com/ajax/statuses/mymblog?uid={UID}&page={PAGE}&feature=0'

response_list = []
for i in range(1, TOTAL_PAGES + 1):
    print('Page {}'.format(i))
    url = POSTS_ENDPOINT.format(UID=UID, PAGE=i)
    headers = {
        'Accept': 'application/json, text/javascript, */*; q=0.01',
        'User-Agent': USER_AGENT,
        'Cookie': 'SUB={}'.format(COOKIE_SUB)
    }
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print('Error: {}'.format(response.status_code))
        print('Response: {}'.format(response.text))
        break
    response_list.append(response.json())
    print('Page {} done'.format(i))
    time.sleep(DELAY)

# with open(f'weibo_{UID}_raw.json', 'w') as f:
#     json.dump(response_list, f)

def get_long_text(mblogid):
    url = f'https://weibo.com/ajax/statuses/longtext?id={mblogid}'
    headers = {
        'Accept': 'application/json, text/javascript, */*; q=0.01',
        'User-Agent': USER_AGENT,
        'Cookie': 'SUB={}'.format(COOKIE_SUB)
    }
    response = requests.get(url, headers=headers)
    if response.status_code != 200 or response.json()['ok'] != 1:
        print('Error: {}'.format(response.status_code))
        print('Response: {}'.format(response.text))
        return None
    try:
        return response.json()['data']['longTextContent']
    except KeyError:
        return None

posts = pd.DataFrame(columns=['id', 'created_at', 'posted_by', 'text'])
for page in response_list:
    for post in page['data']['list']:
        if post['user']['id'] != UID:
            continue
        text = post['text_raw']
        if post['isLongText']:
            print(f'Long text: {post["mblogid"]}')
            text = get_long_text(post['mblogid'])
            time.sleep(DELAY)

        posts = pd.concat([posts, pd.DataFrame({
            'id': [post['id']],
            'created_at': [post['created_at']],
            'text': [text]
        })])

posts['created_at'] = pd.to_datetime(posts['created_at'], format='%a %b %d %H:%M:%S %z %Y')

# create folder data if not exists
if not os.path.exists('data'):
    os.makedirs('data')

posts.to_csv('data/posts.csv', index=False)

print('File has been saved as posts.csv')

