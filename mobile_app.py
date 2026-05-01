import os
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
import faiss

app = Flask(__name__)
CORS(app)

# --- GLOBAL MODELS (Lazy Loading) ---
model = None
index = None
df = None

def load_resources():
    global model, index, df
    if model is None:
        print("📥 Loading AI models (this happens once)...")
        # Use a very small, fast model
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Load books
        df = pd.read_csv('book.csv')
        df['combined'] = df['title'].fillna('') + " " + df['authors'].fillna('') + " " + df['description'].fillna('')
        
        # Create FAISS index
        embeddings = model.encode(df['combined'].tolist(), show_progress_bar=False)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings).astype('float32'))
        print("✓ AI Engine initialized locally!")

@app.route('/api/mobile/health', methods=['GET'])
def health():
    return jsonify({
        "status": "active",
        "engine": "Local Sentence-Transformer (Optimized)",
        "message": "AI Engine is Ready! 🚀"
    })

@app.route('/api/mobile/recommend/idea', methods=['POST'])
def recommend_by_idea():
    load_resources() # Ensure loaded
    
    data = request.json
    idea = data.get('idea', '')

    if not idea:
        return jsonify({"error": "No idea provided"}), 400

    try:
        # Vector search
        query_vector = model.encode([idea])
        k = 10
        D, I = index.search(np.array(query_vector).astype('float32'), k)
        
        results = []
        for idx in I[0]:
            if idx < len(df):
                row = df.iloc[idx]
                results.append({
                    "_id": str(row['book_id']),
                    "title": row['title'],
                    "author": row['authors'],
                    "coverUrl": row['image_url'],
                    "description": row['description']
                })

        return jsonify(results)

    except Exception as e:
        print(f"Local AI Error: {e}")
        return jsonify({"error": "AI calculation failed", "details": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
