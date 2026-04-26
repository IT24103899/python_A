import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import joblib

# 1. Load your specific dataset
print("Loading dataset...")
df = pd.read_csv('book.csv')

# 2. Pre-processing
# We only need books that have a description to train the "vibe"
df = df.dropna(subset=['description'])
# Reset index so it matches the matrix rows
df = df.reset_index(drop=True)

# 3. Vectorization (The AI part)
# This converts book descriptions into a mathematical "vibe" map
print("Analyzing book vibes (TF-IDF)...")
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
tfidf_matrix = tfidf.fit_transform(df['description'])

# 4. Training the Recommender
# We use Cosine Similarity to find books with similar descriptions
print("Training the recommendation model...")
model = NearestNeighbors(metric='cosine', algorithm='brute')
model.fit(tfidf_matrix)

# 5. Save the trained model and the data
# We save the dataframe too so the API can return titles and images
joblib.dump({
    'model': model,
    'tfidf': tfidf,
    'dataframe': df
}, 'book_vibe_model.joblib')

print("Success! 'book_vibe_model.joblib' has been created.")
print(f"Model trained on {len(df)} books.")