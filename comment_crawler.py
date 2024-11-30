import requests
import os
import pandas as pd
import time
import sys

if 'COOKIE_SUB' not in os.environ:
    print('Please set the COOKIE_SUB environment variable')
    exit(1)

if len(sys.argv) < 2:
    print('Usage: python comment_crawler.py <file_path>')
    print('No file path provided, using default file path data/posts.csv')

COOKIE_SUB = os.environ['COOKIE_SUB']
FILE_PATH = sys.argv[1] if len(sys.argv) > 1 else 'data/posts.csv'

UID = 1743951792
DELAY = 0.2
TOTAL_PAGES = 1
USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
COMMENTS_ENDPOINT = 'https://weibo.com/ajax/statuses/buildComments?flow=0&is_reload=1&id={ID}&is_show_bulletin=2&is_mix=0&count=100&uid={UID}&fetch_level=0'

def get_comments(post_id):
    url = COMMENTS_ENDPOINT.format(ID=post_id, UID=UID)
    headers = {
        'User-Agent': USER_AGENT,
        'Cookie': 'SUB={}'.format(COOKIE_SUB)
    }
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print('Error: ', response.status_code)
        return None
    return response.json()

posts = pd.read_csv(FILE_PATH)

comments = pd.DataFrame()
for postid in posts['id']:
    print('Getting comments for post', postid)
    pc = get_comments(postid)
    for c in pc['data']:
        comments = pd.concat([comments, pd.DataFrame({
            'post_id': [postid],
            'comment_id': c['id'],
            'text': c['text'],
            'created_at': c['created_at'],
            'user_id': c['user']['id'],
            'user_name': c['user']['screen_name'],
            'user_location': c['user']['location'],
            'user_followers': c['user']['followers_count'],
        })])
    time.sleep(DELAY)

comments['created_at'] = pd.to_datetime(comments['created_at'], format='%a %b %d %H:%M:%S %z %Y')

# create folder data if not exists
if not os.path.exists('data'):
    os.makedirs('data')
    
comments.to_csv('data/comments.csv', index=False)

print('File has been saved as comments.csv')


