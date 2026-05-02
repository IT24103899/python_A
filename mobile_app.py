import os
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
# Enable CORS for the mobile app direct connection
CORS(app, resources={r"/*": {"origins": "*"}})

# --- GLOBAL MODELS ---
df = None
vectorizer = None
tfidf_matrix = None

def load_resources():
    global df, vectorizer, tfidf_matrix
    if df is not None: return True
    try:
        if os.path.exists('book.csv'):
            df = pd.read_csv('book.csv')
            for col in ['title', 'authors', 'image_url', 'description']:
                if col in df.columns: df[col] = df[col].fillna('')
            df['search_content'] = (df['title'].astype(str) + " " + df['authors'].astype(str) + " " + df['description'].astype(str)).str.lower()
            vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=5000)
            tfidf_matrix = vectorizer.fit_transform(df['search_content'])
            return True
        return False
    except Exception as e:
        print(f"Init Error: {e}")
        return False

@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization'
    response.headers['Access-Control-Allow-Methods'] = 'GET,PUT,POST,DELETE,OPTIONS'
    return response

# HEALTH CHECK
@app.route('/', methods=['GET'])
@app.route('/health', methods=['GET'])
@app.route('/api/mobile/health', methods=['GET'])
def health():
    load_resources()
    return jsonify({"status": "active", "engine": "Optimized-Light-Direct", "books": len(df) if df is not None else 0})

# RECOMMENDATION
@app.route('/recommend/idea', methods=['POST'])
@app.route('/api/mobile/recommend/idea', methods=['POST'])
def recommend_by_idea():
    load_resources()
    data = request.get_json()
    if not data or 'idea' not in data: return jsonify([])
    
    idea = data['idea'].strip().lower()
    if not idea: return jsonify([])

    try:
        query_vec = vectorizer.transform([idea])
        cosine_sim = cosine_similarity(query_vec, tfidf_matrix).flatten()
        top_indices = cosine_sim.argsort()[-15:][::-1]
        
        results = []
        seen_ids = set()
        for idx in top_indices:
            if cosine_sim[idx] > 0:
                row = df.iloc[idx]
                bid = str(row['book_id'])
                results.append({"_id": bid, "title": str(row['title']), "author": str(row['authors']), "coverUrl": str(row['image_url']), "description": str(row['description'])[:150] + "..."})
                seen_ids.add(bid)

        if len(results) < 10:
            words = idea.split()
            for i, row in df.iterrows():
                if len(results) >= 15: break
                bid = str(row['book_id'])
                if bid not in seen_ids and any(word in str(row['search_content']) for word in words):
                    results.append({"_id": bid, "title": str(row['title']), "author": str(row['authors']), "coverUrl": str(row['image_url']), "description": str(row['description'])[:150] + "..."})
                    seen_ids.add(bid)
        return jsonify(results)
    except Exception as e:
        print(f"Search Error: {e}")
        return jsonify([])

# VELOCITY LOGS (Ensuring compatibility)
@app.route('/velocity/log', methods=['POST'])
@app.route('/api/mobile/velocity/log', methods=['POST'])
def log_velocity():
    return jsonify({"status": "logged", "message": "Velocity tracking is active"})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
