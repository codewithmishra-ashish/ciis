import pandas as pd
import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from flask import Flask, jsonify
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
from dash import Dash, html, dcc, dash_table, Input, Output
import threading
import time

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
            import json
            json.dump(keywords, f)
        logging.info(f"Keywords updated: {keywords}")
    except Exception as e:
        logging.error(f"Error saving keywords: {str(e)}")

# Task 2: Fetch Tweets (Simulated with Mock Data)
def fetch_tweets(keyword, limit=20):
    tweets = []
    try:
        for i in range(limit):
            content = random.choice([
                f"Sample tweet about {keyword} with some noise",
                f"Urgent: {keyword} is trending! #India",
                f"Neutral tweet mentioning India",
                f"Critical post: {keyword} is a problem!"
            ])
            user = f"user_{random.randint(1, 1000)}"
            tweet_id = f"mock_{i}_{keyword}"
            tweets.append({
                "username": user,
                "tweet": content,
                "retweets_count": random.randint(0, 50),
                "likes_count": random.randint(0, 100),
                "date": datetime.now() - timedelta(minutes=random.randint(0, 1440)),
                "id": tweet_id,
                "link": f"https://mock-x.com/{user}/status/{tweet_id}"
            })
        logging.info(f"Simulated {len(tweets)} tweets for keyword: {keyword}")
    except Exception as e:
        logging.error(f"Error simulating tweets for {keyword}: {str(e)}")
    return tweets

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

# Task 7: Alert System (Flask Endpoint)
flask_app = Flask(__name__)
ENGAGEMENT_THRESHOLD = 15

@flask_app.route("/alerts", methods=["GET"])
def get_alerts():
    try:
        flagged_posts = [analyze_post(row) for _, row in df.iterrows() if analyze_post(row).get("flagged", False)]
        high_engagement = [post for post in flagged_posts if post["engagement"] >= ENGAGEMENT_THRESHOLD]
        alerts = [
            {"user": post["user"], "content": post["content"], "engagement": post["engagement"], 
             "sentiment": post["sentiment"], "link": post["link"]}
            for post in high_engagement
        ]
        logging.info(f"Generated {len(alerts)} alerts")
        return jsonify(alerts)
    except Exception as e:
        logging.error(f"Error generating alerts: {str(e)}")
        return jsonify({"error": "Failed to generate alerts"}), 500

# Task 8: Enhanced Dashboard (Dash)
def generate_dashboard(tweets, G):
    try:
        df = pd.DataFrame(tweets)
        if df.empty:
            print("No data for dashboard")
            logging.info("No data for dashboard")
            return
        
        # Prepare data for table
        table_data = df[["user", "content", "engagement", "sentiment", "link"]].copy()
        table_data["content"] = table_data["content"].str.slice(0, 50) + "..."
        table_data["sentiment"] = table_data["sentiment"].round(2)
        table_data["link"] = table_data["link"].apply(lambda x: f'<a href="{x}" target="_blank">View Post</a>')

        # Engagement trend
        df["date"] = pd.to_datetime(df["date"])
        df["hour"] = df["date"].dt.floor("h")
        engagement_trend = df.groupby("hour")["engagement"].sum().reset_index()

        # Sentiment histogram
        sentiment_hist = df["sentiment"]

        # Network graph
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

        # Dash app
        dash_app = Dash(__name__, server=flask_app, url_base_pathname='/dashboard/')
        
        dash_app.layout = html.Div([
            html.H1("Anti-India Campaign Detection Dashboard", style={'color': '#FFFFFF', 'textAlign': 'center'}),
            html.Label("Filter by Date Range:", style={'color': '#FFFFFF'}),
            dcc.DatePickerRange(
                id='date-picker',
                min_date_allowed=df["date"].min(),
                max_date_allowed=df["date"].max(),
                initial_visible_month=df["date"].max(),
                start_date=df["date"].min(),
                end_date=df["date"].max()
            ),
            html.Br(),
            html.H3("Flagged Posts", style={'color': '#FFFFFF'}),
            dash_table.DataTable(
                id='table',
                data=table_data.to_dict('records'),
                columns=[
                    {"name": "User", "id": "user"},
                    {"name": "Content", "id": "content"},
                    {"name": "Engagement", "id": "engagement"},
                    {"name": "Sentiment", "id": "sentiment"},
                    {"name": "Link", "id": "link", "type": "text", "presentation": "markdown"}
                ],
                style_table={'overflowX': 'auto'},
                style_cell={'backgroundColor': '#1A1A1A', 'color': '#FFFFFF', 'borderColor': '#444444'},
                style_header={'backgroundColor': '#333333', 'color': '#FFFFFF'}
            ),
            html.H3("Engagement Trend", style={'color': '#FFFFFF'}),
            dcc.Graph(
                figure=px.line(
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
            ),
            html.H3("Sentiment Distribution", style={'color': '#FFFFFF'}),
            dcc.Graph(
                figure=px.histogram(
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
            ),
            html.H3("Influencer Network", style={'color': '#FFFFFF'}),
            dcc.Graph(
                figure=go.Figure(data=[edge_trace, node_trace]).update_layout(
                    showlegend=False,
                    plot_bgcolor="#1A1A1A",
                    paper_bgcolor="#1A1A1A",
                    font_color="#FFFFFF"
                )
            )
        ], style={'backgroundColor': '#1A1A1A', 'padding': '20px'})

        # Callback for date filtering
        @dash_app.callback(
            Output('table', 'data'),
            Input('date-picker', 'start_date'),
            Input('date-picker', 'end_date')
        )
        def update_table(start_date, end_date):
            filtered_df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]
            filtered_table = filtered_df[["user", "content", "engagement", "sentiment", "link"]].copy()
            filtered_table["content"] = filtered_table["content"].str.slice(0, 50) + "..."
            filtered_table["sentiment"] = filtered_table["sentiment"].round(2)
            filtered_table["link"] = filtered_table["link"].apply(lambda x: f'<a href="{x}" target="_blank">View Post</a>')
            return filtered_table.to_dict('records')

        # Run Dash app in a separate thread
        dash_thread = threading.Thread(target=lambda: dash_app.run(debug=True, host="0.0.0.0", port=8050, use_reloader=False))
        dash_thread.start()
        logging.info("Dashboard started on port 8050")
        dash_thread.join(timeout=2)  # Allow Dash to initialize briefly

    except Exception as e:
        logging.error(f"Dashboard generation error: {str(e)}")

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
            import json
            json.dump(report, f)
        logging.info("Report generated: report.json")
    except Exception as e:
        logging.error(f"Report generation error: {str(e)}")

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
    
    # Generate dashboard (Task 8)
    generate_dashboard(flagged_posts, G)
    
    # Start Flask app only once in the main thread after a short delay
    time.sleep(1)  # Ensure Dash thread has a chance to start
    logging.info("Starting Flask app on port 5000")
    flask_app.run(debug=True, host="0.0.0.0", port=5000, threaded=True)