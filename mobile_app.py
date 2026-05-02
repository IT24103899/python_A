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

# --- GLOBAL MODELS (Lazy loaded to save memory) ---
df = None
index = None
metadata = None
model = None
vectorizer = None
tfidf_matrix = None

def load_resources():
    global df, index, metadata, model, vectorizer, tfidf_matrix
    
    if df is not None:
        return True

    try:
        print("📥 Initializing AI Engine...")
        
        # 1. Load the main book database
        if os.path.exists('book.csv'):
            df = pd.read_csv('book.csv')
            print(f"✓ Database loaded: {len(df)} books")
        else:
            print("❌ Error: book.csv not found!")
            df = pd.DataFrame()
            return False

        # 2. Try loading Advanced Semantic AI (FAISS)
        if ADVANCED_AI_AVAILABLE and os.path.exists('books.index') and os.path.exists('books_metadata.pkl'):
            print("🧠 Loading Advanced Semantic AI (FAISS)...")
            index = faiss.read_index('books.index')
            with open('books_metadata.pkl', 'rb') as f:
                metadata = pickle.load(f)
            model = SentenceTransformer('all-MiniLM-L6-v2')
            print("✅ Semantic Engine Ready!")
            return True
        
        # 3. Fallback to Lightweight TF-IDF
        print("⚡ Falling back to Lightweight TF-IDF Engine...")
        df['combined'] = (df['title'].fillna('') + " " + df['description'].fillna('')).str.lower()
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(df['combined'])
        print("✅ Light Engine Ready!")
        return True
        
    except Exception as e:
        print(f"❌ Error during AI initialization: {e}")
        return False

# Bulletproof CORS: inject headers on every response
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
        "engine": "Advanced (Semantic)" if index is not None else "Basic (TF-IDF)",
        "message": "AI Service is Healthy"
    })

@app.route('/api/mobile/recommend/idea', methods=['POST'])
def recommend_by_idea():
    load_resources() # Ensure data is ready
    
    data = request.get_json()
    if not data or 'idea' not in data:
        return jsonify({"error": "No idea provided"}), 400
    
    idea = data['idea']
    results = []

    try:
        print(f"🔍 AI Analyzing Idea: {idea}")
        
        if index is not None and model is not None:
            # --- ADVANCED SEMANTIC SEARCH ---
            query_vector = model.encode([idea]).astype('float32')
            distances, indices = index.search(query_vector, 15)
            
            for idx in indices[0]:
                if idx < len(df):
                    row = df.iloc[idx]
                    results.append({
                        "_id": str(row['book_id']),
                        "title": str(row['title']),
                        "author": str(row['authors']),
                        "coverUrl": str(row['image_url']),
                        "description": str(row.get('description', ''))[:150] + "..."
                    })
        else:
            # --- LIGHTWEIGHT TF-IDF FALLBACK ---
            if tfidf_matrix is not None:
                query_vec = vectorizer.transform([idea.lower()])
                cosine_sim = cosine_similarity(query_vec, tfidf_matrix).flatten()
                top_indices = cosine_sim.argsort()[-15:][::-1]
                
                for idx in top_indices:
                    if cosine_sim[idx] > 0.05:
                        row = df.iloc[idx]
                        results.append({
                            "_id": str(row['book_id']),
                            "title": str(row['title']),
                            "author": str(row['authors']),
                            "coverUrl": str(row['image_url']),
                            "description": str(row.get('description', ''))[:150] + "..."
                        })

        print(f"✅ Found {len(results)} matches")
        return jsonify(results)

    except Exception as e:
        print(f"AI Recommendation Error: {e}")
        return jsonify([])

if __name__ == '__main__':
    # Use standard Render PORT
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
