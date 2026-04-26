import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
import os
import math
import numpy as np
from spellchecker import SpellChecker
from textblob import TextBlob
import re

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}}, supports_credentials=True)

print("Starting up the AI-Scholar Recommendation Engine (FAISS Fast Mode)...")
print("CORS is enabled for all origins.")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load Stage 2: Cross-Encoder (for "Hard" Accuracy)
print("Loading Cross-Encoder for advanced re-ranking...")
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

print("Loading FAISS index and metadata...")
try:
    index = faiss.read_index("books.index")
    df_meta = pd.read_pickle("books_metadata.pkl")
    # Ensure description column exists and is filled
    if 'description' not in df_meta.columns:
        df_meta['description'] = ''
    df_meta['description'] = df_meta['description'].fillna('')
    print("FAISS index loaded successfully.")
except Exception as e:
    print("Error loading FAISS index:", e)

print("Loading user book.csv to get the newest info...")
df_csv = pd.read_csv("book.csv")

# Ensure 'id' exists
if 'book_id' in df_csv.columns and 'id' not in df_csv.columns:
    df_csv['id'] = df_csv['book_id']
if 'id' not in df_csv.columns and 'bookId' in df_csv.columns:
    df_csv['id'] = df_csv['bookId']

# Normalize cover column
if 'cover_url' not in df_csv.columns:
    for col in ['image_url', 'images.url', 'image', 'images_url']:
        if col in df_csv.columns:
            df_csv['cover_url'] = df_csv[col]
            break

if 'description' not in df_csv.columns:
    df_csv['description'] = ''
else:
    df_csv['description'] = df_csv['description'].fillna("")

# --- LOAD GRAMMAR TOOL ---
print("Loading Grammar Checker (PySpellChecker + TextBlob)...")
spell = SpellChecker()
print("Grammar Checker Loaded!")

print("Model and Database Loaded! API is active and listening.")

@app.route('/api/recommend/<int:book_id>', methods=['GET'])
def recommend_books(book_id):
    try:
        # A. Get the seed book from user's CSV
        seed_book = df_csv[df_csv['id'] == book_id]
        if seed_book.empty:
            return jsonify({"error": f"Could not find book with ID {book_id} in CSV"}), 404
        
        seed_book_data = seed_book.iloc[0]
        seed_title = seed_book_data['title']
        seed_description = str(seed_book_data.get('description', ''))
        
        # B. Encode seed request
        import numpy as np
        query_vector = model.encode([seed_description]).astype('float32')
        
        # C. Fast Similarity Search!
        D, I = index.search(query_vector, 5) # search top 5 to ensure we get 3 valid Others
        candidate_indices = I[0]
        
        # Extract the original book_id of the matches from metadata
        recommended_ids = df_meta.iloc[candidate_indices]['book_id'].tolist()
        
        # D. Map back to user's updated CSV
        results = []
        for match_id, score in zip(recommended_ids, D[0]):
            if int(match_id) == book_id:
                continue # Skip the seed book itself
            
            # Find row in user's CSV
            row_matches = df_csv[df_csv['id'] == match_id]
            if not row_matches.empty:
                row = row_matches.iloc[0]
                
                # Check for null image
                cover = row.get('cover_url')
                if pd.isna(cover):
                    cover = "https://via.placeholder.com/150x220?text=No+Cover"
                
                # Estimate a match score from 0-100% 
                match_percentage = max(70.0, 100.0 - float(score)*15.0)
                
                results.append({
                    "id": int(row['id']),
                    "title": str(row['title']),
                    "description": str(row.get('description', '')),
                    "cover_url": cover,
                    "match_score": round(match_percentage, 1)
                })
            
            if len(results) >= 3:
                break
                
        return jsonify({
            "status": "success",
            "based_on_book": str(seed_title),
            "recommendations": results
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

from flask import request

@app.route('/api/recommend/text', methods=['POST'])
def recommend_books_text():
    try:
        print("🔵 Received /api/recommend/text request")
        data = request.json
        seed_title = data.get('title', '')
        seed_description = data.get('description', '')
        seed_id = data.get('id', 0)
        print(f"📚 Searching for: {seed_title}")
        
        query_text = seed_title + " " + seed_description
        if not query_text.strip():
            return jsonify({"error": "No title or description provided"}), 400
            
        import numpy as np
        query_vector = model.encode([query_text]).astype('float32')
        
        # Search for top matches
        D, I = index.search(query_vector, 6) # Top 6 to allow filtering
        candidate_indices = I[0]
        recommended_ids = df_meta.iloc[candidate_indices]['book_id'].tolist()
        
        results = []
        for match_id, score in zip(recommended_ids, D[0]):
            if int(match_id) == seed_id:
                continue
                
            row_matches = df_csv[df_csv['id'] == match_id]
            if not row_matches.empty:
                row = row_matches.iloc[0]
                # Filter out exactly matching titles
                if str(row.get('title', '')).lower() == seed_title.lower():
                    continue
                    
                cover = row.get('cover_url')
                if pd.isna(cover): cover = "https://via.placeholder.com/150x220?text=No+Cover"
                match_percentage = max(70.0, 100.0 - float(score)*15.0)
                
                results.append({
                    "id": int(row['id']),
                    "title": str(row['title']),
                    "author": str(row.get('authors', '')),
                    "cover_url": cover,
                    "match_score": round(match_percentage, 1)
                })
                if len(results) >= 3: break
                
        return jsonify({
            "status": "success",
            "based_on_book": seed_title,
            "recommendations": results
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/', methods=['GET'])
def api_index():
    return jsonify({"status": "recommender", "message": "running (FAISS fast mode with advanced re-ranking)"})

# --- ADVANCED CHAT WITH RE-RANKING ---
@app.route('/chat', methods=['POST'])
def advanced_bot():
    try:
        user_query = request.json.get('message', '')

        if not user_query:
            return jsonify({"error": "No message provided"}), 400

        # --- STAGE 1: Retrieval ---
        query_vector = model.encode([user_query]).astype('float32')
        # Get top 30 candidates quickly
        D, I = index.search(query_vector, 30)
        candidate_indices = I[0]
        candidates = df_meta.iloc[candidate_indices].copy()

        # --- STAGE 2: Re-ranking (The "Hard" Part) ---
        # We pair the user query with each candidate description
        # Handle null/NaN descriptions
        pairs = [[user_query, str(desc) if pd.notna(desc) else ""] for desc in candidates['description']]
        # The Cross-Encoder predicts how relevant each pair is
        scores = cross_encoder.predict(pairs)
        candidates['rel_score'] = scores

        # Sort by the new, more accurate score
        top_books = candidates.sort_values(by='rel_score', ascending=False).head(5)

        # --- RESPONSE GENERATION ---
        results = []
        for _, book in top_books.iterrows():
            results.append({
                "title": book['title'],
                "author": book['authors'],
                "image": book['image_url'],
                "summary": book['description'][:150] + "..."
            })

        reply = f"Based on your request, I've analyzed the library and re-ranked these as your best matches:"
        return jsonify({"reply": reply, "books": results})
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# --- SPELLING & GRAMMAR API ENDPOINT ---
@app.route('/api/check-grammar', methods=['POST'])
def check_grammar():
    try:
        # 1. Get the text the user typed in the React form
        from flask import request
        data = request.json
        user_text = data.get('text', '')

        if not user_text:
            return jsonify({"status": "error", "message": "No text provided"}), 400

        mistakes = []
        
        # 2. Check spelling with PySpellChecker
        misspelled = spell.unknown(user_text.split())
        for word in misspelled:
            corrections = spell.correction(word)
            start_idx = user_text.find(word)
            if start_idx != -1:
                mistakes.append({
                    "mistake": word,
                    "message": f"Spelling: '{word}' is misspelled",
                    "suggestions": [corrections] if corrections else [],
                    "offset": start_idx,
                    "length": len(word),
                    "type": "spelling"
                })
        
        # 3. Check grammar with pattern matching
        # Subject-verb agreement: check for common mistakes like "we does", "I am", etc.
        words = user_text.split()
        
        # Pattern 1: Subject + incorrect verb form
        grammar_patterns = [
            (r'\b(we|they|you|we|all)\s+does\b', 'do', 'Subject-verb agreement'),
            (r'\b(he|she|it|john|mary)\s+do\b', 'does', 'Subject-verb agreement'),
            (r'\b(I)\s+am\s+', 'am', 'Correct'),  # I am is correct
            (r'\b(he|she|it)\s+are\b', 'is', 'Subject-verb agreement'),
            (r'\b(we|they|you)\s+is\b', 'are', 'Subject-verb agreement'),
        ]
        
        text_lower = user_text.lower()
        
        for pattern, correction, error_type in grammar_patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                mistake_text = user_text[match.start():match.end()]
                
                # Skip if it's actually correct (like "I am")
                if "correct" not in error_type.lower():
                    # Find the verb that needs correction
                    parts = mistake_text.split()
                    if len(parts) >= 2:
                        incorrect_verb = parts[-1]
                        mistakes.append({
                            "mistake": mistake_text,
                            "message": f"Grammar: {error_type} - Use '{correction}' instead of '{incorrect_verb}'",
                            "suggestions": [mistake_text.replace(incorrect_verb, correction)],
                            "offset": match.start(),
                            "length": match.end() - match.start(),
                            "type": "grammar"
                        })

        return jsonify({
            "status": "success",
            "original_text": user_text,
            "mistakes_found": len(mistakes),
            "details": mistakes
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', '5000'))
    app.run(host='0.0.0.0', port=port, debug=False)