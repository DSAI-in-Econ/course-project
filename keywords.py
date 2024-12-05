import pandas as pd
import jieba.analyse
import re


# 假设你有一列推文文本
posts = pd.read_csv('processed/posts.csv')
posts.head()

# 自定义停用词表
stop_words = set([
    "http", "我们", "30", "2024", "10", "How", "Make", "long", "always",
    "over", "utm", "http", "RT", "NCAA", "Square","cn", "20", "How", "Make", "run"])

tweets=posts['text']

def is_chinese(word):
    return bool(re.search(r"[\u4e00-\u9fa5]", word))

# 过滤非汉字关键词
important_keywords = []
for tweet in tweets:
    keywords = jieba.analyse.extract_tags(tweet, topK=5, allowPOS=('n', 'v'))
    filtered_keywords = [word for word in keywords if is_chinese(word) and word not in stop_words]
    important_keywords.append(filtered_keywords[0] if filtered_keywords else "无关键词")

for i, keyword in enumerate(important_keywords):
    print(f"推文 {i+1} 的关键词: {keyword}")

