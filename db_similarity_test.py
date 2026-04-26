import mysql.connector
from sentence_transformers import SentenceTransformer, util

# --- 1. CONNECT TO YOUR MYSQL DATABASE ---
# CHANGE THESE to match your local MySQL setup!
print("Connecting to database...")
db_connection = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Bharana@2004",
    database="elibrary_db"
)
cursor = db_connection.cursor(dictionary=True)

# --- 2. LOAD THE AI MODEL ---
print("Loading AI Model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# --- 3. GET THE "SEED" BOOK (The one the user just read) ---
# For this test, let's assume the user just read the book with book_id = 1 (e.g., 1984)
target_book_id = 1508

cursor.execute("SELECT title, description FROM books WHERE id = %s", (target_book_id,))
seed_book = cursor.fetchone()

if not seed_book:
    print(f"Error: Could not find book with ID {target_book_id}")
    exit()

print(f"\nUser just read: ** {seed_book['title']} **")
print("Finding similar books...\n")

# --- 4. GET ALL OTHER BOOKS (To recommend) ---
cursor.execute("SELECT id, title, description FROM books WHERE id != %s", (target_book_id,))
unread_books = cursor.fetchall()

# --- 5. RUN THE AI MATH ---
# Convert the seed book description into a vector
seed_embedding = model.encode(seed_book['description'])

# Extract just the descriptions from the unread books list and convert them
unread_descriptions = [book['description'] for book in unread_books]
unread_embeddings = model.encode(unread_descriptions)

# Calculate Cosine Similarity
cosine_scores = util.cos_sim(seed_embedding, unread_embeddings)[0]

# --- 6. ATTACH SCORES AND SORT ---
# Add the AI score to our database results
for i in range(len(unread_books)):
    unread_books[i]['match_score'] = cosine_scores[i].item() * 100

# Sort the books from Highest Score to Lowest Score
recommended_books = sorted(unread_books, key=lambda x: x['match_score'], reverse=True)

# --- 7. PRINT THE TOP 3 RESULTS ---
print("--- TOP 3 RECOMMENDATIONS ---")
for i in range(3): # Only print the top 3
    book = recommended_books[i]
    print(f"{i+1}. {book['title']} (Match: {book['match_score']:.2f}%)")
    print(f"   Description: {book['description'][:75]}...\n")

# Close the database connection
cursor.close()
db_connection.close()