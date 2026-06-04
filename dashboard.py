# ============================================
# SENTIMENT ANALYSIS DASHBOARD
# ============================================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# ---- PAGE CONFIG ----
st.set_page_config(
    page_title="Customer Review Analysis",
    page_icon="📊",
    layout="wide"
)

# ---- LOAD DATA ----
@st.cache_data
def load_data():
    return pd.read_csv('Data/analyzed_reviews.csv')

df = load_data()

# ---- TITLE ----
st.title("📊 Customer Review Sentiment Analysis")
st.markdown("Analysis of **40,000 Amazon customer reviews** using Python & NLP")
st.divider()

# ---- SECTION 1: OVERVIEW METRICS ----
st.header("📈 Overview")

total = len(df)
positive_pct = round(df['sentiment'].value_counts()['Positive'] / total * 100, 1)
negative_pct = round(df['sentiment'].value_counts()['Negative'] / total * 100, 1)
neutral_pct  = round(df['sentiment'].value_counts()['Neutral']  / total * 100, 1)
avg_score    = round(df['compound_score'].mean(), 3)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Reviews",    f"{total:,}")
col2.metric("Positive Reviews", f"{positive_pct}%", "😊")
col3.metric("Negative Reviews", f"{negative_pct}%", "😞")
col4.metric("Avg Sentiment Score", f"{avg_score}")

st.divider()

# ---- SECTION 2: SENTIMENT ANALYSIS ----
st.header("🎭 Sentiment Analysis")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Sentiment Distribution")
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ['#2ecc71', '#e74c3c', '#95a5a6']
    df['sentiment'].value_counts().plot(kind='bar', color=colors, ax=ax)
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Number of Reviews")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    st.pyplot(fig)

with col2:
    st.subheader("Compound Score Distribution")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(df['compound_score'], bins=50, color='steelblue', ax=ax)
    ax.set_xlabel("Compound Score (-1 → +1)")
    ax.set_ylabel("Count")
    st.pyplot(fig)

st.divider()

# ---- SECTION 3: TOPIC ANALYSIS ----
st.header("🔍 Topic Analysis")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Topic Distribution")
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ['#3498db','#2ecc71','#e74c3c','#f39c12','#9b59b6','#95a5a6']
    df['topic'].value_counts().plot(kind='bar', color=colors, ax=ax)
    ax.set_xlabel("Topic")
    ax.set_ylabel("Number of Reviews")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    st.pyplot(fig)

with col2:
    st.subheader("Sentiment by Topic")
    fig, ax = plt.subplots(figsize=(6, 4))
    topic_sentiment = df.groupby(['topic', 'sentiment']).size().unstack()
    topic_sentiment.plot(kind='bar',
                         color=['#e74c3c', '#95a5a6', '#2ecc71'],
                         ax=ax)
    ax.set_xlabel("Topic")
    ax.set_ylabel("Number of Reviews")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.legend(title='Sentiment')
    st.pyplot(fig)

st.divider()

# ---- SECTION 4: WORD CLOUD ----
st.header("☁️ Word Cloud")

text_all = ' '.join(df['cleaned_text'].dropna().tolist())
wordcloud = WordCloud(
    width=1200, height=400,
    background_color='white',
    colormap='Blues',
    max_words=100
).generate(text_all)

fig, ax = plt.subplots(figsize=(12, 4))
ax.imshow(wordcloud, interpolation='bilinear')
ax.axis('off')
st.pyplot(fig)

st.divider()

# ---- SECTION 5: EXPLORE DATA ----
st.header("🔎 Explore Reviews")

col1, col2 = st.columns(2)
with col1:
    sentiment_filter = st.selectbox(
        "Filter by Sentiment",
        ["All", "Positive", "Negative", "Neutral"]
    )
with col2:
    topic_filter = st.selectbox(
        "Filter by Topic",
        ["All"] + df['topic'].unique().tolist()
    )

filtered_df = df.copy()
if sentiment_filter != "All":
    filtered_df = filtered_df[filtered_df['sentiment'] == sentiment_filter]
if topic_filter != "All":
    filtered_df = filtered_df[filtered_df['topic'] == topic_filter]

st.dataframe(
    filtered_df[['Text', 'Score', 'sentiment', 'compound_score', 'topic']]
    .head(50),
    use_container_width=True
)

st.caption(f"Showing {len(filtered_df):,} reviews")