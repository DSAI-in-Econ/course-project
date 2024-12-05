from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import pandas as pd
import re
from nltk.corpus import stopwords
from sklearn.metrics import silhouette_score
import jieba
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# 定义预处理函数
def preprocess_text(text):
    # 去掉非中文字符（保留汉字）
    text = re.sub(r'[^\u4e00-\u9fa5]', '', text)
    # 去掉多余的空格或其他杂乱字符
    text = text.strip()
    return text


# 推文数据
posts = pd.read_csv('processed/posts.csv')

tweets=posts['text']

# 预处理推文
cleaned_tweets = [preprocess_text(tweet) for tweet in tweets]

# 对预处理后的推文进行分词
segmented_tweets = [" ".join(jieba.cut(tweet)) for tweet in cleaned_tweets]

# 定义 TfidfVectorizer
vectorizer = TfidfVectorizer(
    max_df=0.7,  # 去掉在70%以上推文中出现的词
    min_df=2,    # 仅保留至少出现在2篇推文中的词
    max_features=1000  # 限制特征数量，避免过多噪声
)

# 转换为向量表示
X = vectorizer.fit_transform(segmented_tweets)
print("保留的特征词：", vectorizer.get_feature_names_out())

# 测试不同簇数的效果
best_k = 0
best_score = -1
for k in range(2, 10):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X)
    score = silhouette_score(X, labels)
    print(f"簇数 {k}, 轮廓系数: {score}")
    if score > best_score:
        best_score = score
        best_k = k

# 执行 K-Means 聚类
order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names_out()

for i in range(best_k):
    print(f"簇 {i} 的关键词:")
    for idx in order_centroids[i, :10]:  # 每个簇显示前10个关键词
        print(f"- {terms[idx]}")


# 使用最佳簇数重新聚类
kmeans = KMeans(n_clusters=best_k, random_state=42)
labels = kmeans.fit_predict(X)

# 将聚类标签添加到原始数据中
posts['cluster'] = labels

# 查看每个簇的推文数量
print(posts['cluster'].value_counts())

# 分析每个簇的情感得分（假设posts中包含评论平均情感 'sentiment_score'）
if 'topic_score' in posts.columns:
    cluster_sentiment_stats = posts.groupby('cluster')['topic_score'].agg(['mean', 'std', 'count']).reset_index()
    print("各簇情感统计:")
    print(cluster_sentiment_stats)

    # 可视化情感分布
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # 替换为支持中文的字体，例如 SimHei
    plt.rcParams['axes.unicode_minus'] = False   # 避免负号显示问题
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='cluster', y='topic_score', data=posts)
    plt.title("不同簇的评论情感分布")
    plt.xlabel("簇标签")
    plt.ylabel("评论平均情感得分")
    plt.show()

    # 保存带聚类标签的数据
    posts.to_csv('clustered_posts.csv', index=False)


# 计算主题强度
def calculate_theme_strength(text, keywords):
    return sum(1 for word in keywords if word in text)

# 提取每个簇的关键词
cluster_keywords = {}
for i in range(best_k):
    cluster_keywords[i] = [terms[idx] for idx in order_centroids[i, :10]]

# 计算每条推文的主题强度
for cluster, keywords in cluster_keywords.items():
    posts[f'cluster_{cluster}_strength'] = posts['text'].apply(lambda x: calculate_theme_strength(x, keywords))

# 查看主题强度与情感得分的相关性
if 'topic_score' in posts.columns:
    for cluster in cluster_keywords.keys():
        correlation = posts[f'cluster_{cluster}_strength'].corr(posts['topic_score'])
        print(f"簇 {cluster} 的主题强度与情感得分的相关系数: {correlation}")

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# 定义自变量（每个簇的主题强度）和因变量（情感得分）
X = posts[[f'cluster_{cluster}_strength' for cluster in cluster_keywords.keys()]]
y = posts['topic_score']

# 拆分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 建立线性回归模型
model = LinearRegression()
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 模型评估
print("R² 得分:", r2_score(y_test, y_pred))
print("均方误差 (MSE):", mean_squared_error(y_test, y_pred))

# 查看每个特征的回归系数
coefficients = model.coef_
for i, strength in enumerate(cluster_keywords.keys()):
    print(f"簇 {strength} 的主题强度回归系数: {coefficients[i]}")

