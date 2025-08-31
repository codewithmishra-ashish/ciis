import pandas as pd
import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging
from datetime import datetime, timedelta
import random
import os
import json
import streamlit as st
from snscrape.modules.twitter import TwitterSearchScraper

# Setup logging for ethical compliance (Task 9)
logging.basicConfig(filename=os.path.join('audit_log.txt'), level=logging.INFO, 
                    format='%(asctime)s - %(message)s')

# Load NLP & Sentiment (Task 3)
nlp = spacy.load("en_core_web_sm")
analyzer = SentimentIntensityAnalyzer()

# Cache for NLP results to improve performance
nlp_cache = {}

# Dynamic keyword/hashtag list (Task 1)
ANTI_INDIA_KEYWORDS = [
    "boycott india", "FreeKashmir", "Modi fascism", "Hindutva attacks",
    "#AntiIndia", "#FreeKashmir", "#BoycottIndia"
]

# Task 1: Keyword Database (Dynamic)
def save_keywords(keywords, file_path=os.path.join("keywords.json")):
    try:
        with open(file_path, 'w') as f:
            json.dump(keywords, f)
        logging.info(f"Keywords updated: {keywords}")
    except Exception as e:
        logging.error(f"Error saving keywords: {str(e)}")

# Task 2: Fetch Tweets (Using snscrape)
def fetch_tweets(keyword, limit=20):
    try:
        tweets = []
        for i, tweet in enumerate(TwitterSearchScraper(keyword + " lang:en").get_items()):
            if i >= limit:
                break
            tweets.append({
                "username": tweet.user.username,
                "tweet": tweet.rawContent,
                "retweets_count": tweet.retweetCount,
                "likes_count": tweet.likeCount,
                "date": tweet.date,
                "id": tweet.id,
                "link": tweet.url
            })
        logging.info(f"Fetched {len(tweets)} tweets for keyword: {keyword}")
        return tweets
    except Exception as e:
        logging.error(f"Error fetching tweets for {keyword}: {str(e)}")
        return []

# Collect tweets
all_posts = []
for kw in ANTI_INDIA_KEYWORDS:
    all_posts.extend(fetch_tweets(kw))
df = pd.DataFrame(all_posts)

# Task 3: NLP and Sentiment Analysis
def analyze_nlp(texts):
    try:
        results = []
        uncached_texts = [t for t in texts if t not in nlp_cache]
        if uncached_texts:
            docs = list(nlp.pipe(uncached_texts))
            for text, doc in zip(uncached_texts, docs):
                entities = [ent.text for ent in doc.ents if ent.label_ in ["GPE", "ORG", "PERSON"]]
                sentiment = analyzer.polarity_scores(text)
                nlp_cache[text] = {"entities": entities, "sentiment": sentiment["compound"]}
        return [nlp_cache[text] for text in texts]
    except Exception as e:
        logging.error(f"NLP error: {str(e)}")
        return [{"entities": [], "sentiment": 0.0} for _ in texts]

# Task 4: Engagement Analysis
def analyze_post(row):
    try:
        text = row.get("tweet", "")
        text_lower = text.lower()
        engagement = row.get("retweets_count", 0) + row.get("likes_count", 0)
        nlp_res = analyze_nlp([text])[0]
        return {
            "user": row.get("username", ""),
            "content": text,
            "retweets": row.get("retweets_count", 0),
            "likes": row.get("likes_count", 0),
            "date": row.get("date", ""),
            "flagged": any(keyword.lower() in text_lower for keyword in ANTI_INDIA_KEYWORDS),
            "sentiment": nlp_res["sentiment"],
            "engagement": engagement,
            "entities": nlp_res["entities"],
            "link": row.get("link", "")
        }
    except Exception as e:
        logging.error(f"Error analyzing post: {str(e)}")
        return {}

# Task 5: Identify Key Influencers and Networks
def build_network(tweets):
    G = nx.DiGraph()
    try:
        for tweet in tweets:
            user = tweet["user"]
            engagement = tweet["engagement"]
            G.add_node(user, engagement=engagement)
            for ent in tweet["entities"]:
                if ent != user:
                    G.add_edge(user, ent, weight=1)
        centrality = nx.degree_centrality(G)
        influencers = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:5]
        logging.info(f"Top influencers: {influencers}")
        return G, influencers
    except Exception as e:
        logging.error(f"Network analysis error: {str(e)}")
        return nx.DiGraph(), []

# Task 6: Detect Coordinated Campaigns
def detect_coordinated_campaigns(tweets):
    try:
        texts = [t["content"] for t in tweets]
        if len(texts) < 2:
            return []
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(texts)
        similarity_matrix = cosine_similarity(tfidf_matrix)
        coordinated = []
        for i in range(len(tweets)):
            for j in range(i + 1, len(tweets)):
                time_diff = abs((tweets[i]["date"] - tweets[j]["date"]).total_seconds())
                if similarity_matrix[i][j] > 0.8 and time_diff < 3600:
                    coordinated.append((tweets[i], tweets[j]))
        logging.info(f"Detected {len(coordinated)} coordinated pairs")
        return coordinated
    except Exception as e:
        logging.error(f"Coordinated campaign detection error: {str(e)}")
        return []

# Task 10: Countermeasure Report
def generate_report(tweets, network, influencers):
    try:
        report = {
            "total_posts": len(tweets),
            "top_influencers": influencers,
            "coordinated_campaigns": len(detect_coordinated_campaigns(tweets)),
            "recommendations": "Monitor top influencers; investigate coordinated posts."
        }
        with open(os.path.join("report.json"), "w") as f:
            json.dump(report, f)
        logging.info("Report generated: report.json")
    except Exception as e:
        logging.error(f"Report generation error: {str(e)}")

# Task 8: Enhanced Dashboard (Streamlit)
def generate_dashboard(tweets, G):
    try:
        df = pd.DataFrame(tweets)
        if df.empty:
            st.write("No data for dashboard")
            logging.info("No data for dashboard")
            return
        
        st.title("Anti-India Campaign Detection Dashboard")

        # Date filter
        df["date"] = pd.to_datetime(df["date"])
        min_date = df["date"].min().date()
        max_date = df["date"].max().date()
        start_date, end_date = st.date_input(
            "Select Date Range",
            [min_date, max_date],
            min_value=min_date,
            max_value=max_date
        )
        filtered_df = df[(df["date"].dt.date >= start_date) & (df["date"].dt.date <= end_date)]

        # Table
        st.subheader("Flagged Posts")
        table_data = filtered_df[["user", "content", "engagement", "sentiment", "link"]].copy()
        table_data["content"] = table_data["content"].str.slice(0, 50) + "..."
        table_data["sentiment"] = table_data["sentiment"].round(2)
        table_data["link"] = table_data["link"].apply(lambda x: f'<a href="{x}" target="_blank">View Post</a>')
        st.dataframe(table_data, use_container_width=True)

        # Engagement trend
        df["hour"] = df["date"].dt.floor("h")
        engagement_trend = df.groupby("hour")["engagement"].sum().reset_index()
        st.subheader("Engagement Trend")
        fig = px.line(
            engagement_trend,
            x="hour",
            y="engagement",
            title="Engagement Trend for Flagged Posts",
            labels={"hour": "Time", "engagement": "Total Engagement (Likes + Retweets)"},
            color_discrete_sequence=["#FF6B6B"]
        ).update_layout(
            plot_bgcolor="#1A1A1A",
            paper_bgcolor="#1A1A1A",
            font_color="#FFFFFF"
        )
        st.plotly_chart(fig)

        # Sentiment histogram
        st.subheader("Sentiment Distribution")
        fig = px.histogram(
            df,
            x="sentiment",
            title="Sentiment Distribution of Flagged Posts",
            labels={"sentiment": "Sentiment Score", "count": "Number of Posts"},
            color_discrete_sequence=["#4ECDC4"],
            nbins=20
        ).update_layout(
            plot_bgcolor="#1A1A1A",
            paper_bgcolor="#1A1A1A",
            font_color="#FFFFFF"
        )
        st.plotly_chart(fig)

        # Network graph
        st.subheader("Influencer Network")
        pos = nx.spring_layout(G)
        edge_x, edge_y = [], []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        edge_trace = go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(color="#FF6B6B"))
        node_x, node_y = [], []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
        node_trace = go.Scatter(x=node_x, y=node_y, mode='markers+text', text=list(G.nodes()), 
                                marker=dict(size=10, color="#4ECDC4"))
        fig = go.Figure(data=[edge_trace, node_trace]).update_layout(
            showlegend=False,
            plot_bgcolor="#1A1A1A",
            paper_bgcolor="#1A1A1A",
            font_color="#FFFFFF"
        )
        st.plotly_chart(fig)

        logging.info("Streamlit dashboard rendered")
    except Exception as e:
        logging.error(f"Dashboard generation error: {str(e)}")
        st.error("Error rendering dashboard")

# Main Execution
if __name__ == "__main__":
    # Save keywords
    save_keywords(ANTI_INDIA_KEYWORDS)

    # Process posts
    flagged_posts = [analyze_post(row) for _, row in df.iterrows() if analyze_post(row).get("flagged", False)]
    
    # Network analysis (Task 5)
    G, influencers = build_network(flagged_posts)
    
    # Coordinated campaign detection (Task 6)
    coordinated = detect_coordinated_campaigns(flagged_posts)
    
    # Generate report
    generate_report(flagged_posts, G, influencers)
    
    # Generate dashboard (Task 8)
    generate_dashboard(flagged_posts, G)