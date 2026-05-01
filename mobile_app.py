import os
import pandas as pd
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# --- CONFIGURATION ---
GEMINI_API_KEY = "AIzaSyBXfzVlohPOGg9Pzh33nEDq8hJlqy1lWcI"
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1/models/gemini-pro:generateContent?key={GEMINI_API_KEY}"

# Load dataset once
try:
    df = pd.read_csv('book.csv')
    df['clean_title'] = df['title'].str.strip()
    available_titles = df['clean_title'].tolist()
    print(f"✓ Loaded {len(df)} books for AI lookup")
except Exception as e:
    print(f"Error loading book.csv: {e}")
    df = pd.DataFrame()
    available_titles = []

@app.route('/api/mobile/health', methods=['GET'])
def health():
    return jsonify({
        "status": "active",
        "engine": "Google Gemini 1.5 Flash (Direct v1)",
        "books_indexed": len(df),
        "message": "AI Engine is Ready and Connected! 🚀"
    })

@app.route('/api/mobile/recommend/idea', methods=['POST'])
def recommend_by_idea():
    data = request.json
    idea = data.get('idea', '')

    if not idea:
        return jsonify({"error": "No idea provided"}), 400

    try:
        prompt = f"""
        You are an expert librarian for an E-Library app. 
        User's interest: "{idea}"
        
        Task: Pick the 5 most relevant books from our catalog that match this interest.
        Return ONLY the exact titles of the books, one per line. No extra text.
        
        Our Catalog (Titles):
        {", ".join(available_titles[:250])} 
        """

        # Direct REST API call to force v1 version
        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }]
        }
        
        response = requests.post(GEMINI_URL, json=payload, timeout=20)
        response_data = response.json()

        if response.status_code != 200:
            print(f"Gemini API Error: {response_data}")
            return jsonify({"error": "AI API Error", "details": response_data}), response.status_code

        # Extract text from Gemini response
        raw_text = response_data['candidates'][0]['content']['parts'][0]['text']
        suggested_titles = [line.strip() for line in raw_text.split('\n') if line.strip()]

        results = []
        for title in suggested_titles:
            # Clean matching
            match = df[df['clean_title'].str.contains(title, case=False, na=False)].head(1)
            if not match.empty:
                results.append({
                    "_id": str(match.iloc[0]['book_id']),
                    "title": match.iloc[0]['title'],
                    "author": match.iloc[0]['authors'],
                    "coverUrl": match.iloc[0]['image_url'],
                    "description": match.iloc[0]['description']
                })

        # Fallback keyword matching
        if len(results) < 2:
            keyword_matches = df[df['title'].str.contains(idea.split()[0], case=False, na=False)].head(5)
            for _, row in keyword_matches.iterrows():
                if not any(r['_id'] == str(row['book_id']) for r in results):
                    results.append({
                        "_id": str(row['book_id']),
                        "title": row['title'],
                        "author": row['authors'],
                        "coverUrl": row['image_url'],
                        "description": row['description']
                    })

        return jsonify(results[:10])

    except Exception as e:
        print(f"Server Error: {e}")
        return jsonify({"error": "Server busy", "details": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
