import streamlit as st
import tweepy
import networkx as nx
import spacy
import pandas as pd
import sqlite3
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from datetime import datetime
from collections import defaultdict
import os
from dotenv import load_dotenv
import time

# Load environment variables for API keys
load_dotenv()

# API credentials from .env
consumer_key = os.getenv("TWITTER_CONSUMER_KEY", "your_consumer_key")
consumer_secret = os.getenv("TWITTER_CONSUMER_SECRET", "your_consumer_secret")
access_token = os.getenv("TWITTER_ACCESS_TOKEN", "your_access_token")
access_token_secret = os.getenv("TWITTER_ACCESS_TOKEN_SECRET", "your_token_secret")
bearer_token = os.getenv("TWITTER_BEARER_TOKEN", None)

# Mock data function for testing without API
def get_mock_data(query, count=100):
    data = []
    for i in range(count):
        data.append({
            'id': i,
            'user_id': f"user_{i % 10}",
            'text': f"Mock post about {query} #{query} {'anti-India' if i % 3 == 0 else ''}",
            'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'retweet_count': i % 5,
            'user_followers': 100 if i % 10 < 7 else 10000
        })
    return data

# Fetch tweets using v2 API with Bearer Token and rate limit handling
def fetch_tweets(query, count=100):
    if bearer_token:
        client = tweepy.Client(bearer_token=bearer_token)
    else:
        client = tweepy.Client(
            consumer_key=consumer_key,
            consumer_secret=consumer_secret,
            access_token=access_token,
            access_token_secret=access_token_secret
        )
    print(f"Client created with {'Bearer Token' if bearer_token else 'OAuth 1.0a'}")
    try:
        tweets = client.search_recent_tweets(
            query=query,
            max_results=min(count, 100),
            tweet_fields=["created_at", "text", "public_metrics"],
            user_fields=["id", "username", "public_metrics"]
        )
        print(f"Tweets response: {tweets}")
        return [
            {
                'id': t.id,
                'user_id': t.author_id,
                'text': t.text,
                'created_at': t.created_at.strftime("%Y-%m-%d %H:%M:%S") if t.created_at else None,
                'retweet_count': t.public_metrics.get('retweet_count', 0),
                'user_followers': next((u.public_metrics.get('followers_count', 0) for u in tweets.includes.get('users', []) if u.id == t.author_id), 0)
            }
            for t in tweets.data or []
            if t.author_id is not None
        ]
    except tweepy.TooManyRequests as e:
        st.warning(f"Rate limit exceeded: {e}. Waiting 15 minutes or use mock data. Using mock data now.")
        time.sleep(900)
        return get_mock_data(query, count)
    except tweepy.TweepyException as e:
        st.warning(f"Error fetching tweets from X API: {e}. Using mock data.")
        return get_mock_data(query, count)

# Initialize NLP
nlp = spacy.load("en_core_web_sm")

# Initialize database
conn = sqlite3.connect("campaigns.db")
cursor = conn.cursor()
cursor.execute('''CREATE TABLE IF NOT EXISTS posts
                 (id TEXT, user_id TEXT, text TEXT, created_at TEXT, retweet_count INTEGER, user_followers INTEGER)''')
conn.commit()

def store_posts(posts):
    for post in posts:
        cursor.execute("INSERT OR REPLACE INTO posts VALUES (?, ?, ?, ?, ?, ?)",
                       (post['id'], post['user_id'], post['text'], post['created_at'],
                        post['retweet_count'], post['user_followers']))
    conn.commit()

def build_interaction_graph(posts):
    G = nx.DiGraph()
    for post in posts:
        user = post['user_id']
        followers = post['user_followers']
        if user is not None and followers is not None:
            G.add_node(user, followers=followers)
            for other_post in posts:
                if (other_post['user_id'] is not None and other_post['user_followers'] is not None and
                    post['id'] != other_post['id'] and abs((datetime.strptime(post['created_at'], "%Y-%m-%d %H:%M:%S") -
                                                          datetime.strptime(other_post['created_at'], "%Y-%m-%d %H:%M:%S")).total_seconds()) < 600):
                    G.add_edge(user, other_post['user_id'], weight=1)
    return G

def analyze_sentiment(text):
    doc = nlp(text)
    entities = [ent.text.lower() for ent in doc.ents if ent.label_ in ["GPE", "NORP"]]
    negative_keywords = ["anti-india", "hate", "oppression", "tyranny", "terrorism"]
    score = sum(1 for kw in negative_keywords if kw in text.lower()) * 0.2
    if "india" in entities:
        score += 0.3
    return min(score, 1.0)

def detect_anomalies(posts):
    df = pd.DataFrame(posts)
    # Ensure required columns exist with default values
    if 'retweet_count' not in df.columns:
        df['retweet_count'] = 0
    if 'user_followers' not in df.columns:
        df['user_followers'] = 0
    features = df[['retweet_count', 'user_followers']].values
    model = IsolationForest(contamination=0.1, random_state=42)
    labels = model.fit_predict(features)
    return [1 if label == -1 else 0 for label in labels]

def calculate_risk_score(posts, graph, anomalies):
    coordination_score = nx.average_clustering(graph) * 0.4
    sentiment_scores = [analyze_sentiment(post['text']) for post in posts]
    sentiment_avg = sum(sentiment_scores) / len(sentiment_scores) * 0.3
    virality_score = sum(post['retweet_count'] for post in posts) / len(posts) * 0.2
    anomaly_score = sum(anomalies) / len(anomalies) * 0.1
    return min((coordination_score + sentiment_avg + virality_score + anomaly_score) * 100, 100)

def visualize_graph(G):
    pos = nx.spring_layout(G)
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines')
    node_x, node_y = [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
    node_trace = go.Scatter(x=node_x, y=node_y, mode='markers', hoverinfo='text',
                            marker=dict(size=[G.nodes[n]['followers'] / 1000 for n in G.nodes()], color='skyblue'))
    return go.Figure(data=[edge_trace, node_trace], layout=go.Layout(showlegend=False, hovermode='closest'))

def main():
    st.title("Anti-India Campaign Detector")
    query = st.text_input("Enter hashtag or keyword (e.g., #IndiaOut)", "#IndiaOut")
    count = st.slider("Number of posts to analyze", 10, 100, 50)
    
    if st.button("Analyze"):
        with st.spinner("Fetching and analyzing data..."):
            posts = fetch_tweets(query, count)
            store_posts(posts)
            graph = build_interaction_graph(posts)
            anomalies = detect_anomalies(posts)
            risk_score = calculate_risk_score(posts, graph, anomalies)
            
            st.subheader("Results")
            st.write(f"Risk Score: {risk_score:.2f}/100")
            if risk_score > 80:
                st.error("High-risk campaign detected!")
            elif risk_score > 50:
                st.warning("Moderate-risk campaign detected.")
            else:
                st.success("Low-risk campaign detected.")
                
            st.subheader("User Interaction Graph")
            fig = visualize_graph(graph)
            st.plotly_chart(fig)
            
            st.subheader("Post Details")
            df = pd.DataFrame(posts)
            df['sentiment'] = [analyze_sentiment(post['text']) for post in posts]
            df['is_anomaly'] = anomalies
            st.dataframe(df[['user_id', 'text', 'created_at', 'retweet_count', 'sentiment', 'is_anomaly']])
            
            st.subheader("Export Report")
            if st.button("Generate CSV"):
                df.to_csv("campaign_report.csv", index=False)
                st.success("Report saved as campaign_report.csv")

if __name__ == "__main__":
    main()