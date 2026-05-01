#!/usr/bin/env python3
"""
AI-Scholar Recommendation Engine
Provides ML-powered book recommendations using FAISS vector search
"""

import os
import sys

# Windows console encoding fix
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
import traceback
from flask import Flask, jsonify, request
from flask_cors import CORS
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
import language_tool_python

# ============================================
# INITIALIZATION
# ============================================
print("=" * 60)
print("Starting AI-Scholar Recommendation Engine...")
print("=" * 60)

app = Flask(__name__)

# Enable CORS for all routes
CORS(app,
     origins="*",
     methods=["GET", "POST", "OPTIONS"],
     allow_headers=["Content-Type", "Authorization"],
     supports_credentials=False)

# Bulletproof CORS: inject headers on EVERY response (handles 404s and preflight too)
@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    return response

# Handle OPTIONS preflight globally for any path
@app.route('/', defaults={'path': ''}, methods=['OPTIONS'])
@app.route('/<path:path>', methods=['OPTIONS'])
def handle_options(path):
    return '', 204

print("✓ CORS enabled for all origins")

# ============================================
# LOAD MODELS
# ============================================
try:
    print("\n📝 Loading Grammar Checker...")
    grammar_tool = language_tool_python.LanguageTool('en-US')
    print("✓ Grammar Checker loaded")
except Exception as e:
    print(f"✗ Error loading Grammar Checker: {e}")
    print("  (Tip: Run 'pip install language-tool-python' if it's missing)")
    grammar_tool = None

try:
    print("\n📦 Loading Sentence Transformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("✓ Sentence Transformer loaded")
except Exception as e:
    print(f"✗ Error loading Sentence Transformer: {e}")
    model = None

try:
    print("\n🔄 Loading Cross-Encoder for re-ranking...")
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    print("✓ Cross-Encoder loaded")
except Exception as e:
    print(f"✗ Error loading Cross-Encoder: {e}")
    cross_encoder = None

# ============================================
# LOAD DATABASE
# ============================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    print("\n🗃️  Loading FAISS index...")
    faiss_index = faiss.read_index(os.path.join(BASE_DIR, "books.index"))
    df_meta = pd.read_pickle(os.path.join(BASE_DIR, "books_metadata.pkl"))
    print(f"✓ FAISS index loaded ({len(df_meta)} books)")
except Exception as e:
    print(f"✗ Error loading FAISS index: {e}")
    faiss_index = None
    df_meta = None

try:
    print("\n📖 Loading book data from CSV...")
    df_csv = pd.read_csv(os.path.join(BASE_DIR, "book.csv"))
    print(f"✓ Book CSV loaded ({len(df_csv)} books)")
    
    # Normalize ID column
    if 'book_id' in df_csv.columns and 'id' not in df_csv.columns:
        df_csv['id'] = df_csv['book_id']
    elif 'bookId' in df_csv.columns and 'id' not in df_csv.columns:
        df_csv['id'] = df_csv['bookId']
    
    # Normalize cover URL column
    if 'cover_url' not in df_csv.columns:
        for col in ['image_url', 'images_url', 'image', 'cover']:
            if col in df_csv.columns:
                df_csv['cover_url'] = df_csv[col]
                break
    
    # Ensure description exists
    if 'description' not in df_csv.columns:
        df_csv['description'] = ''
    df_csv['description'] = df_csv['description'].fillna('')
    
    # Normalize author column
    if 'author' not in df_csv.columns:
        if 'authors' in df_csv.columns:
            df_csv['author'] = df_csv['authors']
        else:
            df_csv['author'] = 'Unknown'
    df_csv['author'] = df_csv['author'].fillna('Unknown')
    
except Exception as e:
    print(f"✗ Error loading CSV: {e}")
    df_csv = None

print("\n" + "=" * 60)
print("✅ AI Engine Ready!")
print("=" * 60 + "\n")

# ============================================
# HELPER FUNCTIONS
# ============================================
def get_book_from_csv(book_id):
    """Get book details from CSV by ID"""
    if df_csv is None or df_csv.empty:
        return None
    matches = df_csv[df_csv['id'] == book_id]
    return matches.iloc[0] if not matches.empty else None

def get_openlibrary_cover(title, author=''):
    """Build an Open Library cover search URL based on title + author.
    Falls back to a placeholder if title is empty."""
    try:
        import urllib.parse
        clean_title = str(title).strip()
        clean_author = str(author).strip()
        if not clean_title:
            return "https://via.placeholder.com/150x220?text=No+Cover"
        # Open Library search → redirect to cover image
        query = f"title={urllib.parse.quote(clean_title)}"
        if clean_author:
            query += f"&author={urllib.parse.quote(clean_author)}"
        return f"https://openlibrary.org/search/covers.json?{query}&limit=1"
    except Exception:
        return "https://via.placeholder.com/150x220?text=No+Cover"

def format_recommendation(db_row, match_score=0.0):
    """Format a book row into recommendation JSON"""
    try:
        import urllib.parse
        title = str(db_row.get('title', 'Unknown'))
        author = str(db_row.get('author', db_row.get('authors', 'Unknown')))

        # Prefer the actual cover URL from the dataset (Goodreads, etc.)
        # Fall back to Open Library title-based URL only if no real URL exists
        raw_cover = (
            db_row.get('image_url') or
            db_row.get('cover_url') or
            db_row.get('coverUrl') or
            db_row.get('images_url') or
            ''
        )
        raw_cover = str(raw_cover).strip()
        if raw_cover and raw_cover.startswith('http'):
            cover = raw_cover
        else:
            cover = f"https://covers.openlibrary.org/b/title/{urllib.parse.quote(title)}-M.jpg"

        return {
            "id": int(db_row.get('id', 0)),
            "title": title,
            "author": author,
            "cover_url": cover,
            "match_score": round(float(match_score), 1),
            "description": str(db_row.get('description', ''))[:200]  # Limit to 200 chars
        }
    except Exception as e:
        print(f"Error formatting recommendation: {e}")
        return None

# ============================================
# HEALTH CHECK
# ============================================
@app.route('/api/health', methods=['GET', 'OPTIONS'])
def health_check():
    """Check if API is healthy"""
    if request.method == 'OPTIONS':
        return '', 204
    
    return jsonify({
        "status": "ok",
        "service": "AI-Scholar Recommendation Engine",
        "model_loaded": model is not None,
        "index_loaded": faiss_index is not None,
        "csv_loaded": df_csv is not None,
        "grammar_loaded": grammar_tool is not None
    }), 200

# ============================================
# RECOMMENDATIONS BY TEXT
# ============================================
@app.route('/api/recommend/text', methods=['POST', 'OPTIONS'])
def recommend_by_text():
    """
    Get recommendations based on book title and description
    Request: {title: string, description: string, id: number}
    """
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        data = request.json or {}
        user_query = str(data.get('query', '')).strip()
        title = str(data.get('title', '')).strip()
        description = str(data.get('description', '')).strip()
        seed_id = int(data.get('id', 0))
        
        # Extract advanced filters
        min_rating = float(data.get('min_rating', 0))
        from_year = int(data.get('from_year')) if data.get('from_year') else 0
        to_year = int(data.get('to_year')) if data.get('to_year') else 3000
        author_filter = str(data.get('author', '')).strip().lower()

        print(f"📋 Request: query='{user_query}' | title='{title[:30]}...' | rating>={min_rating} | years={from_year}-{to_year}")
        
        # Validate input
        query_text = user_query if user_query else f"{title} {description}".strip()
        if not model or faiss_index is None:
            return jsonify({
                "status": "error",
                "message": "Model or index not loaded",
                "recommendations": []
            }), 500
            
        # If no strict query text is provided, do a pure filter search over the dataset
        if not query_text:
            results = []
            # We simply iterate over the dataset and find the first 20 that match the filters
            for idx in range(len(df_meta)):
                if len(results) >= 20:
                    break
                
                meta_row = df_meta.iloc[idx]
                book_id = int(meta_row.get('book_id', 0))
                
                if book_id == seed_id:
                    continue
                    
                csv_row = get_book_from_csv(book_id)
                if csv_row is None:
                    continue
                    
                # --- APPLY ADVANCED FILTERS ---
                if author_filter:
                    book_author = str(csv_row.get('authors', csv_row.get('author', ''))).lower()
                    if author_filter not in book_author:
                        continue
                
                if min_rating > 0:
                    book_rating = csv_row.get('average_rating', 0)
                    try:
                        book_rating_val = float(book_rating) if pd.notna(book_rating) else 0.0
                        if book_rating_val < min_rating:
                            continue
                    except (ValueError, TypeError):
                        continue
                
                if from_year > 0 or to_year < 3000:
                    pub_year = csv_row.get('original_publication_year')
                    try:
                        if pd.notna(pub_year):
                            book_year = int(float(pub_year))
                            if not (from_year <= book_year <= to_year):
                                continue
                    except (ValueError, TypeError):
                        pass
                # -------------------------------
                
                match_score = 100.0  # Perfect match for pure filtering
                rec = format_recommendation(csv_row, match_score)
                if rec:
                    results.append(rec)
                    
            print(f"📚 Returning {len(results)} recommendations (Pure Filter)")
            return jsonify({
                "status": "success",
                "based_on_book": "Filters Only",
                "recommendations": results
            }), 200
        
        # Encode and search
        try:
            query_vector = model.encode([query_text]).astype('float32')
            distances, indices = faiss_index.search(query_vector, 100)  # Get top 100, filter later
        except Exception as e:
            print(f"Error during search: {e}")
            return jsonify({
                "status": "error",
                "message": "Search failed",
                "recommendations": []
            }), 500
        
        # Build recommendations
        results = []
        for match_id, distance in zip(indices[0], distances[0]):
            if len(results) >= 20:
                break
            
            try:
                # Get from metadata
                meta_row = df_meta.iloc[match_id]
                book_id = int(meta_row.get('book_id', 0))
                
                # Skip if same as seed
                if book_id == seed_id:
                    continue
                
                # Get full details from CSV
                csv_row = get_book_from_csv(book_id)
                if csv_row is None:
                    continue
                
                # Skip if exact title match
                if str(csv_row.get('title', '')).lower() == title.lower():
                    continue
                
                # --- APPLY ADVANCED FILTERS ---
                # Author
                if author_filter:
                    book_author = str(csv_row.get('authors', csv_row.get('author', ''))).lower()
                    if author_filter not in book_author:
                        continue
                
                # Minimum Rating
                if min_rating > 0:
                    book_rating = csv_row.get('average_rating', 0)
                    try:
                        book_rating_val = float(book_rating) if pd.notna(book_rating) else 0.0
                        if book_rating_val < min_rating:
                            continue
                    except (ValueError, TypeError):
                        continue
                
                # Publication Year
                if from_year > 0 or to_year < 3000:
                    pub_year = csv_row.get('original_publication_year')
                    try:
                        if pd.notna(pub_year):
                            book_year = int(float(pub_year))
                            if not (from_year <= book_year <= to_year):
                                continue
                    except (ValueError, TypeError):
                        pass # Ignore books with invalid year formats and let them through
                # -------------------------------
                
                # Calculate match score
                match_score = max(70.0, 100.0 - float(distance) * 12.0)
                
                rec = format_recommendation(csv_row, match_score)
                if rec:
                    results.append(rec)
            except Exception as e:
                print(f"Error processing result: {e}")
                continue
        
        print(f"📚 Returning {len(results)} recommendations")
        return jsonify({
            "status": "success",
            "based_on_book": title,
            "recommendations": results
        }), 200
    
    except Exception as e:
        print(f"❌ Error in recommend/text: {e}")
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": str(e),
            "recommendations": []
        }), 500

# ============================================
# RECOMMENDATIONS BY BOOK ID
# ============================================
@app.route('/api/recommend/<int:book_id>', methods=['GET', 'OPTIONS'])
def recommend_by_id(book_id):
    """Get recommendations for a specific book"""
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        print(f"🔖 Request: recommendations for book ID {book_id}")
        
        # Get seed book
        seed_book = get_book_from_csv(book_id)
        if seed_book is None:
            return jsonify({
                "status": "error",
                "message": f"Book {book_id} not found",
                "recommendations": []
            }), 404
        
        title = str(seed_book.get('title', ''))
        description = str(seed_book.get('description', ''))
        
        # Use text recommendation logic
        query_text = f"{title} {description}".strip()
        
        if not model or faiss_index is None:
            return jsonify({
                "status": "error",
                "message": "Model not loaded",
                "recommendations": []
            }), 500
        
        # Search
        query_vector = model.encode([query_text]).astype('float32')
        distances, indices = faiss_index.search(query_vector, 10)
        
        results = []
        for match_id, distance in zip(indices[0], distances[0]):
            if len(results) >= 3:
                break
            
            meta_row = df_meta.iloc[match_id]
            matched_book_id = int(meta_row.get('book_id', 0))
            
            if matched_book_id == book_id:
                continue
            
            csv_row = get_book_from_csv(matched_book_id)
            if csv_row is None:
                continue
            
            match_score = max(70.0, 100.0 - float(distance) * 12.0)
            rec = format_recommendation(csv_row, match_score)
            if rec:
                results.append(rec)
        
        print(f"📚 Returning {len(results)} recommendations for '{title}'")
        return jsonify({
            "status": "success",
            "based_on_book": title,
            "recommendations": results
        }), 200
    
    except Exception as e:
        print(f"❌ Error in recommend/id: {e}")
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": str(e),
            "recommendations": []
        }), 500

# ============================================
# SPELLING & GRAMMAR
# ============================================
@app.route('/api/check-grammar', methods=['POST', 'OPTIONS'])
def check_grammar():
    """Check grammar and spelling for a given text"""
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        data = request.json or {}
        user_text = data.get('text', '')
        
        if not user_text:
            return jsonify({"status": "error", "message": "No text provided"}), 400
            
        if grammar_tool is None:
            return jsonify({"status": "error", "message": "Grammar tool is not loaded. Please contact administrator."}), 503
            
        # Run grammar check
        matches = grammar_tool.check(user_text)
        
        mistakes = []
        for match in matches:
            error_len = getattr(match, 'errorLength', getattr(match, 'error_length', 0))
            mistakes.append({
                "mistake": user_text[match.offset : match.offset + error_len],
                "message": match.message,
                "suggestions": match.replacements[:3],
                "offset": match.offset,
                "length": error_len
            })
            
        return jsonify({
            "status": "success",
            "original_text": user_text,
            "mistakes_found": len(mistakes),
            "details": mistakes
        }), 200
        
    except Exception as e:
        print(f"❌ Error in check-grammar: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

# ============================================
# ROOT ENDPOINT
# ============================================
@app.route('/', methods=['GET'])
def index():
    """Root endpoint"""
    return jsonify({
        "service": "AI-Scholar Recommendation Engine",
        "status": "running",
        "version": "2.0",
        "endpoints": [
            "GET /api/health",
            "POST /api/recommend/text",
            "GET /api/recommend/<book_id>",
            "POST /api/check-grammar"
        ]
    }), 200

@app.route('/api/routes', methods=['GET'])
def list_routes():
    """Debug endpoint to list all registered routes"""
    routes = []
    for rule in app.url_map.iter_rules():
        if rule.endpoint != 'static':
            routes.append(str(rule))
    return jsonify({"routes": routes}), 200

# ============================================
# READING VELOCITY & ANALYTICS ENDPOINTS
# ============================================
print("[DEBUG] Attempting to import velocity analyzer...")
try:
    from reading_velocity import ReadingVelocityAnalyzer
    print("[DEBUG] ✓ Successfully imported ReadingVelocityAnalyzer")
    
    # Initialize analyzer
    velocity_analyzer = ReadingVelocityAnalyzer()
    print(f"[DEBUG] ✓ Successfully initialized velocity analyzer: {velocity_analyzer}")
except Exception as e:
    print(f"[ERROR] Failed to import/initialize velocity analyzer: {e}")
    import traceback
    traceback.print_exc()
    velocity_analyzer = None

@app.route('/api/velocity/log-session', methods=['POST', 'OPTIONS'])
def log_reading_session():
    """Log a reading session with pages and duration"""
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        data = request.get_json()
        user_id = data.get('userId')
        book_id = data.get('bookId')
        pages_read = data.get('pagesRead', 0)
        duration_seconds = data.get('durationSeconds', 0)
        
        if not all([user_id, book_id]):
            return jsonify({"status": "error", "message": "Missing required fields"}), 400
        
        session = velocity_analyzer.log_reading_session(
            user_id, book_id, pages_read, duration_seconds
        )
        
        return jsonify({
            "status": "success",
            "session": session
        }), 201
    
    except Exception as e:
        print(f"Error logging session: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/velocity/calculate/<int:user_id>/<int:book_id>', methods=['GET', 'OPTIONS'])
def get_velocity(user_id, book_id):
    """Calculate velocity metrics for a user-book combination"""
    print(f"[DEBUG] get_velocity called: user={user_id}, book={book_id}")
    if request.method == 'OPTIONS':
        return '', 204
    
    if velocity_analyzer is None:
        return jsonify({"status": "error", "message": "Velocity analyzer not initialized"}), 500
    
    try:
        velocity = velocity_analyzer.calculate_velocity(user_id, book_id)
        print(f"[DEBUG] velocity result: {velocity}")
        
        if "error" in velocity:
            return jsonify({
                "status": "error",
                "message": velocity["error"]
            }), 404
        
        return jsonify({
            "status": "success",
            "data": velocity
        }), 200
    
    except Exception as e:
        print(f"[ERROR] Error calculating velocity: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/velocity/estimate-completion', methods=['POST', 'OPTIONS'])
def estimate_completion():
    """Estimate time to complete a book based on current velocity"""
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        data = request.get_json()
        user_id = data.get('userId')
        book_id = data.get('bookId')
        total_pages = data.get('totalPages', 0)
        current_page = data.get('currentPage', 0)
        
        if not all([user_id, book_id, total_pages]):
            return jsonify({"status": "error", "message": "Missing required fields"}), 400
        
        estimate = velocity_analyzer.estimate_completion(
            user_id, book_id, total_pages, current_page
        )
        
        if "error" in estimate:
            return jsonify({
                "status": "error",
                "message": estimate["error"]
            }), 400
        
        return jsonify({
            "status": "success",
            "data": estimate
        }), 200
    
    except Exception as e:
        print(f"Error estimating completion: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/velocity/heatmap/<int:user_id>', methods=['GET', 'OPTIONS'])
def get_reading_heatmap(user_id):
    """Get reading activity heatmap for past N days"""
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        days = request.args.get('days', 28, type=int)
        
        heatmap = velocity_analyzer.get_reading_heatmap(user_id, days)
        
        return jsonify({
            "status": "success",
            "data": heatmap
        }), 200
    
    except Exception as e:
        print(f"Error generating heatmap: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/velocity/user-stats/<int:user_id>', methods=['GET', 'OPTIONS'])
def get_user_stats(user_id):
    """Get comprehensive reading statistics for a user"""
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        stats = velocity_analyzer.get_reading_stats(user_id)
        
        if "error" in stats:
            return jsonify({
                "status": "error",
                "message": stats["error"]
            }), 404
        
        return jsonify({
            "status": "success",
            "data": stats
        }), 200
    
    except Exception as e:
        print(f"Error getting user stats: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/velocity/time-of-day/<int:user_id>', methods=['GET', 'OPTIONS'])
def get_time_of_day_analytics(user_id):
    """Analyze reading patterns by time of day (morning, afternoon, evening, night)"""
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        analytics = velocity_analyzer.get_time_of_day_analytics(user_id)
        
        if "error" in analytics:
            return jsonify({
                "status": "error",
                "message": analytics["error"]
            }), 404
        
        return jsonify({
            "status": "success",
            "data": analytics
        }), 200
    
    except Exception as e:
        print(f"Error getting time of day analytics: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/velocity/timeline/<int:user_id>', methods=['GET', 'OPTIONS'])
@app.route('/api/velocity/timeline/<int:user_id>/<int:book_id>', methods=['GET', 'OPTIONS'])
def get_session_timeline(user_id, book_id=None):
    """Get detailed chronological timeline of reading sessions with timestamps"""
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        timeline = velocity_analyzer.get_session_timeline(user_id, book_id)
        
        if "error" in timeline:
            return jsonify({
                "status": "error",
                "message": timeline["error"]
            }), 404
        
        return jsonify({
            "status": "success",
            "data": timeline
        }), 200
    
    except Exception as e:
        print(f"Error getting session timeline: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


# ============================================
# PERSONALIZED RECOMMENDATIONS (NEW AI PART)
# ============================================
@app.route('/api/recommend/personal', methods=['POST', 'OPTIONS'])
def recommend_personalized():
    """
    NEW: AI-Powered 'Books for You' Engine
    Logic: If history is provided, find centroid of taste. If not, suggest top picks.
    """
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        data = request.json or {}
        liked_ids = data.get('bookIds', [])
        limit = int(data.get('limit', 12))
        
        print(f"🧠 Personalizing results for {len(liked_ids)} books (limit={limit})...")
        
        # Scenario A: User has shared history (Personalized)
        if liked_ids and faiss_index is not None and df_meta is not None:
            vectors = []
            valid_ids = []
            
            # 1. Collect embeddings for history
            for bid in liked_ids:
                try:
                    # Find index in metadata where book_id matches
                    match = df_meta[df_meta['book_id'] == int(bid)]
                    if not match.empty:
                        # Reconstruct vector or look up if stored? 
                        # Since vectors aren't stored in metadata (only in index), 
                        # we either re-encode the title+desc or index lookup is needed.
                        # For simplicity, we'll find the title in df_csv and re-encode.
                        csv_row = get_book_from_csv(int(bid))
                        if csv_row is not None:
                            txt = f"{csv_row.get('title', '')} {csv_row.get('description', '')}".strip()
                            vec = model.encode([txt])
                            vectors.append(vec[0])
                            valid_ids.append(int(bid))
                except Exception:
                    continue
            
            if len(vectors) > 0:
                # 2. Calculate taste centroid (average vector)
                centroid = np.mean(vectors, axis=0).astype('float32').reshape(1, -1)
                
                # 3. Search nearest neighbors
                # We fetch extra to filter out the liked books
                distances, indices = faiss_index.search(centroid, limit + len(liked_ids) + 5)
                
                results = []
                for match_idx, dist in zip(indices[0], distances[0]):
                    if len(results) >= limit:
                        break
                    
                    try:
                        meta_row = df_meta.iloc[match_idx]
                        mid = int(meta_row.get('book_id', 0))
                        
                        # Skip if already liked
                        if mid in valid_ids:
                            continue
                            
                        csv_row = get_book_from_csv(mid)
                        if csv_row is not None:
                            # Dynamic score
                            match_score = max(65.0, 98.0 - float(dist) * 10.0)
                            rec = format_recommendation(csv_row, match_score)
                            if rec:
                                results.append(rec)
                    except:
                        continue
                
                return jsonify({
                    "status": "success",
                    "type": "personalized",
                    "recommendations": results
                }), 200

        # Scenario B: No history or index empty (Discovery Mode)
        if df_csv is not None and not df_csv.empty:
            # Filter for high quality books
            top_picks = df_csv[df_csv['average_rating'] >= 4.0]
            
            # Sort by rating count to ensure quality
            if 'ratings_count' in top_picks.columns:
                top_picks = top_picks.sort_values(by='ratings_count', ascending=False)
            
            # Take top 100 and sample randomly for variety
            sample_pool = top_picks.head(100)
            discovery_list = sample_pool.sample(min(limit, len(sample_pool))).to_dict('records')
            
            results = []
            for row in discovery_list:
                rec = format_recommendation(row, 90.0 + (np.random.random() * 5))
                if rec:
                    results.append(rec)
                    
            return jsonify({
                "status": "success",
                "type": "discovery",
                "recommendations": results
            }), 200

        return jsonify({"status": "error", "message": "No data available"}), 404

    except Exception as e:
        print(f"❌ Error in personalization: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

# ============================================
# ERROR HANDLERS
# ============================================
@app.errorhandler(404)
def not_found(error):
    return jsonify({"status": "error", "message": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"status": "error", "message": "Internal server error"}), 500

# ============================================
# RUN SERVER
# ============================================
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"🚀 Starting Flask server on http://0.0.0.0:{port}")
    print("Press CTRL+C to stop\n")
    app.run(
        host='0.0.0.0',
        port=port,
        debug=False,
        threaded=True
    )
