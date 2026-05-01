import os
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)

# --- GLOBAL MODELS (Pre-loaded & Ultra-Light) ---
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = None
df = None

def load_resources():
    global tfidf_matrix, df
    try:
        print("📥 Loading Light-AI Engine...")
        df = pd.read_csv('book.csv')
        df['combined'] = (
            df['title'].fillna('') + " " + 
            df['authors'].fillna('') + " " + 
            df['description'].fillna('')
        ).str.lower()
        
        # This is near-instant for 1,200 books
        tfidf_matrix = vectorizer.fit_transform(df['combined'])
        print(f"✓ AI Engine Ready! ({len(df)} books indexed)")
    except Exception as e:
        print(f"Error loading book.csv: {e}")
        df = pd.DataFrame()

# Load immediately on startup (Safe because it's light)
load_resources()

@app.route('/api/mobile/health', methods=['GET'])
def health():
    return jsonify({
        "status": "active",
        "engine": "Light-AI (TF-IDF)",
        "message": "AI Engine is Instant and Ready! ⚡"
    })

@app.route('/api/mobile/recommend/idea', methods=['POST'])
def recommend_by_idea():
    data = request.json
    idea = data.get('idea', '')

    if not idea:
        return jsonify({"error": "No idea provided"}), 400

    try:
        print(f"🔍 AI searching for: {idea}")
        # Transform query and calculate similarity
        query_vec = vectorizer.transform([idea.lower()])
        cosine_sim = cosine_similarity(query_vec, tfidf_matrix).flatten()
        
        # Get top indices
        top_indices = cosine_sim.argsort()[-15:][::-1]
        
        results = []
        for idx in top_indices:
            score = cosine_sim[idx]
            if score > 0.05: # Reasonable threshold
                row = df.iloc[idx]
                results.append({
                    "_id": str(row['book_id']),
                    "title": row['title'],
                    "author": row['authors'],
                    "coverUrl": row['image_url'],
                    "description": row['description'],
                    "score": float(score)
                })

        # Fallback: If AI results are low, do a direct keyword search
        if len(results) < 3:
            keywords = idea.lower().split()
            for kw in keywords:
                if len(kw) < 3: continue
                matches = df[df['title'].str.lower().str.contains(kw, na=False)].head(5)
                for _, row in matches.iterrows():
                    # Avoid duplicates
                    if not any(r['_id'] == str(row['book_id']) for r in results):
                        results.append({
                            "_id": str(row['book_id']),
                            "title": row['title'],
                            "author": row['authors'],
                            "coverUrl": row['image_url'],
                            "description": row['description'],
                            "score": 0.0
                        })

        print(f"✅ Found {len(results)} matches")
        return jsonify(results[:15])

    except Exception as e:
        print(f"AI Error: {e}")
        return jsonify({"error": "AI calculation failed", "details": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
