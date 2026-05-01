import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import joblib
import numpy as np
import os
import shutil

print("1. Backing up original dataset...")
if not os.path.exists("book_full.csv"):
    shutil.copy("book.csv", "book_full.csv")
    print("Backed up to book_full.csv")
else:
    print("Backup book_full.csv already exists.")

print("\n2. Loading and shrinking data...")
# Load the original data
df = pd.read_csv('book_full.csv')
print(f"Original size: {len(df)} books")

# Drop rows without description first
df = df.dropna(subset=['description']).reset_index(drop=True)

# Take 25% of the data (approx 1000-1200 books)
df_small = df.sample(frac=0.25, random_state=42).reset_index(drop=True)
print(f"Shrunk size: {len(df_small)} books")

# Save the smaller CSV
df_small.to_csv("book.csv", index=False)
print("Saved smaller book.csv")

print("\n3. Re-building FAISS Index and Metadata...")
# Use the same model
print("Initializing Bi-Encoder...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Create Vector Embeddings
print("Encoding books into vectors...")
descriptions = df_small['description'].tolist()
embeddings = model.encode(descriptions, show_progress_bar=True)

# Build FAISS Index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings).astype('float32'))

# Save everything
faiss.write_index(index, "books.index")
df_small.to_pickle("books_metadata.pkl")

print("\nSUCCESS: Shrunk dataset and re-created index!")
print(f"CSV file is now ~{os.path.getsize('book.csv') / 1024 / 1024:.2f} MB")
print(f"Index file is now ~{os.path.getsize('books.index') / 1024 / 1024:.2f} MB")
print(f"Metadata file is now ~{os.path.getsize('books_metadata.pkl') / 1024 / 1024:.2f} MB")
