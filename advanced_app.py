from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer, CrossEncoder
import numpy as np
from spellchecker import SpellChecker

app = Flask(__name__)
CORS(app)

# Load Stage 1: Bi-Encoder (for speed)
bi_encoder = SentenceTransformer('all-MiniLM-L6-v2')
# Load Stage 2: Cross-Encoder (for "Hard" Accuracy)
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# Load the AI Database
index = faiss.read_index("books.index")
df = pd.read_pickle("books_metadata.pkl")

# --- LOAD GRAMMAR TOOL ---
print("Loading Grammar Checker (PySpellChecker)...")
spell = SpellChecker()
print("Grammar Checker Loaded!")


@app.route('/chat', methods=['POST'])
def advanced_bot():
    user_query = request.json.get('message', '')

    # --- STAGE 1: Retrieval ---
    query_vector = bi_encoder.encode([user_query]).astype('float32')
    # Get top 30 candidates quickly
    D, I = index.search(query_vector, 30)
    candidate_indices = I[0]
    candidates = df.iloc[candidate_indices].copy()

    # --- STAGE 2: Re-ranking (The "Hard" Part) ---
    # We pair the user query with each candidate description
    pairs = [[user_query, desc] for desc in candidates['description']]
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


# --- SPELLING & GRAMMAR API ENDPOINT ---
@app.route('/api/check-grammar', methods=['POST'])
def check_grammar():
    try:
        # 1. Get the text the user typed in the React form
        data = request.json
        user_text = data.get('text', '')

        if not user_text:
            return jsonify({"status": "error", "message": "No text provided"}), 400

        # 2. Use PySpellChecker for accurate spell checking
        misspelled = spell.unknown(user_text.split())
        mistakes = []
        
        # Check for spelling errors
        for word in misspelled:
            corrections = spell.correction(word)
            start_idx = user_text.find(word)
            if start_idx != -1:
                mistakes.append({
                    "mistake": word,
                    "message": f"Spelling: '{word}' is misspelled",
                    "suggestions": [corrections] if corrections else [],
                    "offset": start_idx,
                    "length": len(word)
                })

        return jsonify({
            "status": "success",
            "original_text": user_text,
            "mistakes_found": len(mistakes),
            "details": mistakes
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Add your HTML code here (same as before)

if __name__ == '__main__':
    app.run(port=5000, debug=False)