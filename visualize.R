library(tidyverse)

# Load the data
posts <- read_csv("processed/posts.csv")
comments <- read_csv("processed/comments.csv")

# Create a new column with the number of comments
posts <- posts %>%
  mutate(num_comments = map_int(id, ~sum(comments$post_id == .)))

# Create a histogram of the number of comments
ggplot(posts, aes(num_comments)) +
  geom_histogram(binwidth = 1, fill = "skyblue", color = "black") +
  labs(title = "Number of Comments per Post",
       x = "Number of Comments",
       y = "Frequency") +
  theme_minimal()

# Save the plot as a PNG file
ggsave("plots/comments_histogram.png", width = 6, height = 4, dpi = 300) 

# Show sorted topic distribution (45 degrees rotation)
posts_sorted_by_topic <- posts %>%
  count(topic) %>%
  arrange(desc(n))

ggplot(posts_sorted_by_topic, aes(x = reorder(topic, n), y = n)) +
  geom_bar(stat = "identity", fill = "skyblue", color = "black") +
  labs(title = "Topic Distribution",
       x = "Topic",
       y = "Number of Posts") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))


# Save the plot as a PNG file
ggsave("plots/topic_distribution.png", width = 6, height = 4, dpi = 300)

# Average sentiment score of the comments per post
comments <- comments %>%
  mutate(sentiment_score = ifelse(sentiment == "positive", 1, ifelse(sentiment == "negative", -1, 0)))
comments_avg_sentiment <- comments %>%
  group_by(post_id) %>%
  summarize(avg_sentiment = mean(sentiment_score, na.rm = TRUE))

# Plot the distribution of average sentiment scores by the post topics (sorted by average sentiment score)
comments_avg_sentiment <- comments_avg_sentiment %>%
  left_join(posts, by = c("post_id" = "id"))

ggplot(comments_avg_sentiment, aes(x = reorder(topic, avg_sentiment), y = avg_sentiment)) +
  geom_boxplot(fill = "skyblue", color = "black") +
  labs(title = "Sentiment Score by Topic",
       x = "Topic",
       y = "Sentiment Score") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Save the plot as a PNG file
ggsave("plots/avg_sentiment_by_topic.png", width = 6, height = 4, dpi = 300) 

