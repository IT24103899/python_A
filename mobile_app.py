#!/usr/bin/env python3
import os
import sys
import pandas as pd
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
import faiss
from reading_velocity import ReadingVelocityAnalyzer
import traceback

# Windows console encoding fix
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

app = Flask(__name__)
CORS(app)

import torch
# CRITICAL: Limit memory usage for Render Free Tier
torch.set_num_threads(1)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# ============================================
# INITIALIZE AI MODELS & DATA
# ============================================
print("--- Initializing Mobile AI Engine (LITE MODE) ---")

model = None
faiss_index = None
df_meta = None
df_csv = None
velocity_analyzer = None
INIT_ERROR = None

try:
    # Check if data files exist
    if not os.path.exists("books.index"):
        raise FileNotFoundError("books.index not found. Make sure data files are present.")
    if not os.path.exists("books_metadata.pkl"):
        raise FileNotFoundError("books_metadata.pkl not found. Make sure data files are present.")
    if not os.path.exists("book.csv"):
        raise FileNotFoundError("book.csv not found. Make sure data files are present.")
    
    # 1. Load Search Index & Metadata FIRST (smaller)
    print("Loading FAISS index...")
    faiss_index = faiss.read_index("books.index")
    print("✓ FAISS index loaded")
    
    print("Loading metadata pickle...")
    df_meta = pd.read_pickle("books_metadata.pkl")
    print("✓ Metadata loaded")
    
    # 2. Load CSV with ONLY necessary columns to save RAM
    print("Loading book CSV with selective columns...")
    needed_cols = ['id', 'book_id', 'title', 'author', 'authors', 'image_url', 'cover_url', 'description', 'category', 'genre']
    available_cols = pd.read_csv("book.csv", nrows=0).columns.tolist()
    use_cols = [c for c in needed_cols if c in available_cols]
    
    df_csv = pd.read_csv("book.csv", usecols=use_cols)
    print(f"✓ Book CSV loaded ({len(df_csv)} books)")
    
    # Normalize CSV columns
    if 'book_id' in df_csv.columns and 'id' not in df_csv.columns:
        df_csv['id'] = df_csv['book_id']
    
    # 3. Load Semantic Search Model LAST
    print("Loading SentenceTransformer (this takes ~300MB RAM)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("✓ SentenceTransformer loaded")
    
    # Initialize Reading Velocity Engine
    print("Initializing Reading Velocity Engine...")
    velocity_analyzer = ReadingVelocityAnalyzer()
    print("✓ Reading Velocity Engine initialized")
    
    print("✓✓✓ All models and datasets loaded successfully ✓✓✓")
    
except Exception as e:
    INIT_ERROR = str(e)
    print(f"✗ Initialization error: {e}")
    print(f"Traceback:\n{traceback.format_exc()}")
    model = None
    faiss_index = None
    df_meta = None
    df_csv = None

# ============================================
# HELPER FUNCTIONS
# ============================================
def get_book_details(book_id):
    if df_csv is None or df_csv.empty:
        return None
    # Handle both string and int IDs
    try:
        bid = int(book_id)
        matches = df_csv[df_csv['id'] == bid]
    except:
        matches = df_csv[df_csv['id'].astype(str) == str(book_id)]
        
    if not matches.empty:
        row = matches.iloc[0]
        return {
            "_id": str(row.get('id', 0)),
            "title": str(row.get('title', 'Unknown')),
            "author": str(row.get('author', row.get('authors', 'Unknown'))),
            "coverUrl": str(row.get('image_url', row.get('cover_url', ''))),
            "category": str(row.get('category', row.get('genre', ''))),
            "description": str(row.get('description', ''))[:150] + "..."
        }
    return None

# ============================================
# ENDPOINTS
# ============================================

# 0. Health Check
@app.route('/api/mobile/health', methods=['GET'])
def health():
    if INIT_ERROR:
        return jsonify({
            "status": "error",
            "service": "Mobile AI Engine",
            "error": INIT_ERROR,
            "models_loaded": False
        }), 500
    
    return jsonify({
        "status": "active",
        "service": "Mobile AI Engine",
        "models_loaded": model is not None and faiss_index is not None,
        "message": "✓ All systems operational"
    }), 200

# 1. Recommendation by IDEA (Semantic Search)
@app.route('/api/mobile/recommend/idea', methods=['POST'])
def recommend_idea():
    """Find 10 books based on user's natural language idea"""
    data = request.json or {}
    idea = data.get('idea', '').strip()
    
    if not idea:
        return jsonify({"error": "No idea provided"}), 400
    
    if not model or faiss_index is None:
        return jsonify({
            "error": "AI Engine models not loaded",
            "details": INIT_ERROR,
            "message": "System is initializing or data files are missing"
        }), 503

    try:
        # 1. Encode the idea into a vector
        query_vector = model.encode([idea]).astype('float32')
        
        # 2. Search the index for top 20 (to allow filtering of duplicates/seed)
        distances, indices = faiss_index.search(query_vector, 20)
        
        # 3. Resolve results to book details
        recommendations = []
        seen_ids = set()
        
        for idx in indices[0]:
            if len(recommendations) >= 10:
                break
                
            meta_row = df_meta.iloc[idx]
            book_id = int(meta_row.get('book_id', 0))
            
            if book_id in seen_ids:
                continue
                
            details = get_book_details(book_id)
            if details:
                recommendations.append(details)
                seen_ids.add(book_id)
                
        return jsonify({
            "status": "success",
            "idea": idea,
            "count": len(recommendations),
            "recommendations": recommendations
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# 2. Reading Velocity: Log Progress
@app.route('/api/mobile/velocity/log', methods=['POST'])
def log_velocity():
    data = request.json or {}
    user_id = data.get('userId')
    book_id = data.get('bookId')
    pages_read = data.get('pagesRead', 0)
    duration_seconds = data.get('durationSeconds', 0)
    
    if not all([user_id, book_id]):
        return jsonify({"error": "Missing identification"}), 400
        
    session = velocity_analyzer.log_reading_session(
        user_id, book_id, pages_read, duration_seconds
    )
    
    return jsonify({"status": "success", "session": session}), 200

# 3. Reading Velocity: Get Stats
@app.route('/api/mobile/velocity/stats/<string:user_id>/<string:book_id>', methods=['GET'])
def get_stats(user_id, book_id):
    stats = velocity_analyzer.calculate_velocity(user_id, book_id)
    if "error" in stats:
        return jsonify({"status": "empty", "message": stats["error"]}), 200
        
    return jsonify({"status": "success", "data": stats}), 200

if __name__ == '__main__':
    # Use the PORT environment variable if available (for Render/Heroku deployment)
    port = int(os.environ.get('PORT', 5001))
    # Running on 0.0.0.0 to be accessible from outside the container
    app.run(host='0.0.0.0', port=port)
