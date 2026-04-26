import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import joblib
import numpy as np

# 1. Load and Clean Data
df = pd.read_csv('book.csv')
df = df.dropna(subset=['description']).reset_index(drop=True)

# 2. Use a high-quality Bi-Encoder
print("Initializing Bi-Encoder...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# 3. Create Vector Embeddings
print("Encoding 4700+ books into vectors...")
descriptions = df['description'].tolist()
embeddings = model.encode(descriptions, show_progress_bar=True)

# 4. Build FAISS Index (The Advanced AI Database)
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings).astype('float32'));

# 5. Save everything
faiss.write_index(index, "books.index")
df.to_pickle("books_metadata.pkl")
print("Advanced FAISS Index Created Successfully!")