import os
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import pickle

# Advanced AI Imports
try:
    import faiss
    from sentence_transformers import SentenceTransformer
    ADVANCED_AI_AVAILABLE = True
except ImportError:
    ADVANCED_AI_AVAILABLE = False

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# --- GLOBAL MODELS ---
df = None
index = None
model = None
vectorizer = None
tfidf_matrix = None

def load_resources():
    global df, index, model, vectorizer, tfidf_matrix
    
    if df is not None:
        return True

    try:
        # 1. Load the main book database
        if os.path.exists('book.csv'):
            df = pd.read_csv('book.csv')
            print(f"✓ Database loaded: {len(df)} books")
            
            # Clean dataframe to prevent JSON errors
            for col in ['title', 'authors', 'image_url', 'description']:
                if col in df.columns:
                    df[col] = df[col].fillna('Unknown')
        else:
            print("❌ Error: book.csv not found!")
            df = pd.DataFrame()
            return False

        # 2. Try loading Advanced Semantic AI (FAISS)
        if ADVANCED_AI_AVAILABLE and os.path.exists('books.index'):
            print("🧠 Loading Advanced Semantic AI...")
            index = faiss.read_index('books.index')
            model = SentenceTransformer('all-MiniLM-L6-v2')
            print("✅ Semantic Engine Ready!")
        
        # 3. Always prepare TF-IDF as a robust fallback
        print("⚡ Preparing TF-IDF Fallback Engine...")
        # Create a rich text field for searching
        df['search_text'] = (
            df['title'].astype(str) + " " + 
            df['authors'].astype(str) + " " + 
            df.get('description', pd.Series(['']*len(df))).astype(str)
        ).str.lower()
        
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(df['search_text'])
        print("✅ Fallback Engine Ready!")
        return True
        
    except Exception as e:
        print(f"❌ Error during AI initialization: {e}")
        return False

@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization'
    response.headers['Access-Control-Allow-Methods'] = 'GET,PUT,POST,DELETE,OPTIONS'
    return response

@app.route('/api/mobile/health', methods=['GET'])
def health():
    load_resources()
    return jsonify({
        "status": "active",
        "has_data": len(df) > 0 if df is not None else False,
        "engine": "Advanced (Semantic)" if index is not None else "Basic (TF-IDF)"
    })

@app.route('/api/mobile/recommend/idea', methods=['POST'])
def recommend_by_idea():
    load_resources()
    
    data = request.get_json()
    if not data or 'idea' not in data:
        return jsonify([])
    
    idea = data['idea'].strip()
    if not idea:
        return jsonify([])

    results = []
    seen_ids = set()

    try:
        # --- PHASE 1: SEMANTIC SEARCH (ADVANCED) ---
        if index is not None and model is not None:
            print(f"🔍 [Semantic] Idea: {idea}")
            query_vector = model.encode([idea]).astype('float32')
            # Get more candidates than needed to filter safely
            distances, indices = index.search(query_vector, 30)
            
            for idx in indices[0]:
                if 0 <= idx < len(df):
                    row = df.iloc[idx]
                    bid = str(row['book_id'])
                    if bid not in seen_ids:
                        results.append({
                            "_id": bid,
                            "title": str(row['title']),
                            "author": str(row['authors']),
                            "coverUrl": str(row['image_url']),
                            "description": str(row.get('description', ''))[:150] + "..."
                        })
                        seen_ids.add(bid)
                if len(results) >= 15: break

        # --- PHASE 2: TF-IDF FALLBACK (If Semantic failed or returned too few) ---
        if len(results) < 10 and tfidf_matrix is not None:
            print(f"🔍 [Fallback-TFIDF] Idea: {idea}")
            query_vec = vectorizer.transform([idea.lower()])
            cosine_sim = cosine_similarity(query_vec, tfidf_matrix).flatten()
            top_indices = cosine_sim.argsort()[-20:][::-1]
            
            for idx in top_indices:
                if cosine_sim[idx] > 0.01: # Very low threshold to ensure WE FIND SOMETHING
                    row = df.iloc[idx]
                    bid = str(row['book_id'])
                    if bid not in seen_ids:
                        results.append({
                            "_id": bid,
                            "title": str(row['title']),
                            "author": str(row['authors']),
                            "coverUrl": str(row['image_url']),
                            "description": str(row.get('description', ''))[:150] + "..."
                        })
                        seen_ids.add(bid)
                if len(results) >= 15: break

        # --- PHASE 3: KEYWORD BRUTE FORCE (Last resort) ---
        if len(results) == 0:
            print(f"🔍 [Fallback-BruteForce] Idea: {idea}")
            words = idea.lower().split()
            for i, row in df.iterrows():
                text = str(row['search_text'])
                if any(word in text for word in words):
                    bid = str(row['book_id'])
                    if bid not in seen_ids:
                        results.append({
                            "_id": bid,
                            "title": str(row['title']),
                            "author": str(row['authors']),
                            "coverUrl": str(row['image_url']),
                            "description": str(row.get('description', ''))[:150] + "..."
                        })
                        seen_ids.add(bid)
                if len(results) >= 15: break

        print(f"✅ Final Result Count: {len(results)}")
        return jsonify(results)

    except Exception as e:
        print(f"❌ Critical AI Error: {e}")
        return jsonify([])

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
