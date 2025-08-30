import pandas as pd
import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from flask import Flask, jsonify
import networkx as nx
import plotly.express as px
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging
from datetime import datetime, timedelta
import random
import os

# Setup logging for ethical compliance (Task 9)
logging.basicConfig(filename='audit_log.txt', level=logging.INFO, 
                    format='%(asctime)s - %(message)s')

# Load NLP & Sentiment (Task 3)
nlp = spacy.load("en_core_web_sm")
analyzer = SentimentIntensityAnalyzer()

# Dynamic keyword/hashtag list (Task 1)
ANTI_INDIA_KEYWORDS = [
    "boycott india", "FreeKashmir", "Modi fascism", "Hindutva attacks",
    "#AntiIndia", "#FreeKashmir", "#BoycottIndia"
]

# Task 1: Keyword Database (Dynamic)
def save_keywords(keywords, file_path="keywords.json"):
    try:
        with open(file_path, 'w') as f:
            import json
            json.dump(keywords, f)
        logging.info(f"Keywords updated: {keywords}")
    except Exception as e:
        logging.error(f"Error saving keywords: {str(e)}")

# Task 2: Fetch Tweets (Simulated with Mock Data)
def fetch_tweets(keyword, limit=50):
    tweets = []
    try:
        # Simulate tweets with realistic attributes
        for i in range(limit):
            # Randomly include keywords in some tweets
            content = random.choice([
                f"Sample tweet about {keyword} with some noise",
                f"Urgent: {keyword} is trending! #India",
                f"Neutral tweet mentioning India",
                f"Critical post: {keyword} is a problem!"
            ])
            tweets.append({
                "username": f"user_{random.randint(1, 1000)}",
                "tweet": content,
                "retweets_count": random.randint(0, 50),
                "likes_count": random.randint(0, 100),
                "date": datetime.now() - timedelta(minutes=random.randint(0, 1440)),  # Within last 24 hours
                "id": f"mock_{i}_{keyword}"
            })
        logging.info(f"Simulated {len(tweets)} tweets for keyword: {keyword}")
    except Exception as e:
        logging.error(f"Error simulating tweets for {keyword}: {str(e)}")
    return tweets

# Collect tweets
all_posts = []
for kw in ANTI_INDIA_KEYWORDS:
    all_posts.extend(fetch_tweets(kw, limit=50))
df = pd.DataFrame(all_posts)

# Task 3: NLP and Sentiment Analysis
def analyze_nlp(text):
    try:
        doc = nlp(text)
        entities = [ent.text for ent in doc.ents if ent.label_ in ["GPE", "ORG", "PERSON"]]
        sentiment = analyzer.polarity_scores(text)
        return {
            "entities": entities,
            "sentiment": sentiment["compound"]
        }
    except Exception as e:
        logging.error(f"NLP error for text: {text[:50]}... - {str(e)}")
        return {"entities": [], "sentiment": 0.0}

# Task 4: Engagement Analysis
def analyze_post(row):
    try:
        text = row.get("tweet", "")
        text_lower = text.lower()
        engagement = row.get("retweets_count", 0) + row.get("likes_count", 0)
        nlp_res = analyze_nlp(text)
        return {
            "user": row.get("username", ""),
            "content": text,
            "retweets": row.get("retweets_count", 0),
            "likes": row.get("likes_count", 0),
            "date": row.get("date", ""),
            "flagged": any(keyword.lower() in text_lower for keyword in ANTI_INDIA_KEYWORDS),
            "sentiment": nlp_res["sentiment"],
            "engagement": engagement,
            "entities": nlp_res["entities"]
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
            # Add edges for mentions (extracted via NLP)
            for ent in tweet["entities"]:
                if ent != user:  # Avoid self-loops
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
                if similarity_matrix[i][j] > 0.8 and time_diff < 3600:  # Similar posts within 1 hour
                    coordinated.append((tweets[i], tweets[j]))
        logging.info(f"Detected {len(coordinated)} coordinated pairs")
        return coordinated
    except Exception as e:
        logging.error(f"Coordinated campaign detection error: {str(e)}")
        return []

# Task 7: Alert System (Flask Endpoint)
app = Flask(__name__)
ENGAGEMENT_THRESHOLD = 15

@app.route("/alerts", methods=["GET"])
def get_alerts():
    try:
        flagged_posts = [analyze_post(row) for _, row in df.iterrows() if analyze_post(row).get("flagged", False)]
        high_engagement = [post for post in flagged_posts if post["engagement"] >= ENGAGEMENT_THRESHOLD]
        alerts = [
            {"user": post["user"], "content": post["content"], "engagement": post["engagement"], "sentiment": post["sentiment"]}
            for post in high_engagement
        ]
        logging.info(f"Generated {len(alerts)} alerts")
        return jsonify(alerts)
    except Exception as e:
        logging.error(f"Error generating alerts: {str(e)}")
        return jsonify({"error": "Failed to generate alerts"}), 500

# Task 8: Visualization Dashboard (Plotly)
def generate_dashboard(tweets):
    try:
        df = pd.DataFrame(tweets)
        if df.empty:
            print("No data for dashboard")
            logging.info("No data for dashboard")
            return
        # Engagement over time
        df["date"] = pd.to_datetime(df["date"])
        df["hour"] = df["date"].dt.floor("H")
        engagement_trend = df.groupby("hour")["engagement"].sum().reset_index()

        # Create line chart
        fig = px.line(
            engagement_trend,
            x="hour",
            y="engagement",
            title="Engagement Trend for Flagged Posts",
            labels={"hour": "Time", "engagement": "Total Engagement (Likes + Retweets)"},
            color_discrete_sequence=["#FF6B6B"]  # Vibrant color
        )
        fig.update_layout(
            plot_bgcolor="#1A1A1A",  # Dark theme
            paper_bgcolor="#1A1A1A",
            font_color="#FFFFFF"
        )
        fig.show()
        logging.info("Dashboard generated")
        
        # Create chart for sentiment distribution
        fig = px.histogram(
            df,
            x="sentiment",
            title="Sentiment Distribution of Flagged Posts",
            labels={"sentiment": "Sentiment Score", "count": "Number of Posts"},
            color_discrete_sequence=["#4ECDC4"],
            nbins=20
        )
        fig.update_layout(
            plot_bgcolor="#1A1A1A",
            paper_bgcolor="#1A1A1A",
            font_color="#FFFFFF"
        )
        fig.show()
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
        with open("report.json", "w") as f:
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
    generate_dashboard(flagged_posts)
    
    # Generate report (Task 10)
    generate_report(flagged_posts, G, influencers)
    
    # Start Flask app (Task 7)
    app.run(debug=True, host="0.0.0.0", port=5000)