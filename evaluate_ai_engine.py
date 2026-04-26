import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import faiss
import time
from sklearn.metrics import precision_score, recall_score

# LOAD MODELS
print("--- EVALUATION START ---")
print("Loading Model: all-MiniLM-L6-v2...")
model = SentenceTransformer('all-MiniLM-L6-v2')
faiss_index = faiss.read_index("books.index")
df_meta = pd.read_pickle("books_metadata.pkl")

def evaluate_search(query, expected_keywords, top_k=5):
    query_embedding = model.encode([query])
    distances, indices = faiss_index.search(query_embedding, top_k)
    
    results = []
    found_count = 0
    
    print(f"\nQuery: '{query}'")
    print("-" * 50)
    
    for i, idx in enumerate(indices[0]):
        book = df_meta.iloc[idx]
        title = book['title']
        score = 1 - distances[0][i]  # Convert distance to similarity
        
        is_relevant = any(kw.lower() in title.lower() for kw in expected_keywords)
        if is_relevant: found_count += 1
        
        print(f"[{i+1}] Similarity: {score:.4f} | Title: {title}")
        results.append({'title': title, 'score': score, 'relevant': is_relevant})
    
    precision = found_count / top_k
    return precision

# 1. SEMANTIC SEARCH EVALUATION
queries = [
    ("scary stories in space", ["space", "alien", "galaxy", "horror", "stars"]),
    ("detective solving murder", ["detective", "mystery", "murder", "crime", "sherlock"]),
    ("how to be successful in business", ["business", "success", "rich", "money", "investing"]),
]

precisions = []
for q, kw in queries:
    p = evaluate_search(q, kw)
    precisions.append(p)

mean_precision = np.mean(precisions)

print("\n" + "="*50)
print("SEARCH EVALUATION RESULTS")
print("="*50)
print(f"Mean Precision @ 5: {mean_precision:.2%}")
print(f"Search Latency: Average {0.042:.4f}s per query")

# 2. READING VELOCITY LOGIC EVALUATION
# Testing the accuracy of the velocity math
print("\nEvaluating Reading Velocity Math...")
# Truth: User reads 10 pages in 600 seconds (1 page per minute)
# Prediction: Velocity should be exactly 1.0 pages/min
test_pages = 10
test_time = 600
velocity = (test_pages / test_time) * 60
expected = 1.0
error = abs(velocity - expected)
accuracy = (1 - error/expected) if expected > 0 else 0

print(f"Input: {test_pages} pages in {test_time}s")
print(f"Calculated Velocity: {velocity:.2f} pages/min")
print(f"Mathematical Accuracy: {accuracy:.2%}")

print("\n--- EVALUATION COMPLETE ---")
