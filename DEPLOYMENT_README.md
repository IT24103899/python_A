# Python AI Engine for E-Library Mobile App

AI-powered book recommendation system using semantic search and FAISS indexing.

## Features
- **Semantic Search**: Understands user intent through natural language queries
- **FAISS Indexing**: Fast similarity search across 20,000+ books
- **Reading Velocity**: Tracks user reading speed and patterns
- **Lightweight**: Optimized for Render Free Tier (~500MB RAM)

## Setup (Local Development)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Ensure Data Files Exist
Required files in the project root:
- `books.index` - FAISS index (~50MB)
- `books_metadata.pkl` - Metadata pickle (~10MB)
- `book.csv` - Book database (~20MB)

### 3. Run Locally
```bash
python mobile_app.py
```
Server runs on `http://localhost:5001`

### 4. Test Health Check
```bash
curl http://localhost:5001/api/mobile/health
```

## Deployment (Render)

### 1. Create Render Service
- Go to [Render Dashboard](https://dashboard.render.com)
- Click "New" → "Web Service"
- Connect your GitHub repo (https://github.com/IT24103899/python_A)
- Build command: `pip install -r requirements.txt`
- Start command: `python mobile_app.py` (Procfile will auto-detect)

### 2. Environment Variables
No special env vars needed for Free Tier.

### 3. Data Files
**CRITICAL**: Commit these to GitHub (they must be in the repo):
```bash
git add books.index books_metadata.pkl book.csv
git commit -m "Add AI data files"
git push
```

### 4. Deploy
Push to `main` branch and Render will auto-deploy.

## API Endpoints

### Health Check
```
GET /api/mobile/health
```
Response:
```json
{
  "status": "active",
  "service": "Mobile AI Engine",
  "models_loaded": true,
  "message": "✓ All systems operational"
}
```

### Recommend by Idea
```
POST /api/mobile/recommend/idea
Authorization: Bearer <token>
Content-Type: application/json

{
  "idea": "detective mystery love stories"
}
```
Response:
```json
{
  "recommendations": [
    {
      "_id": "123",
      "title": "The Girl in the Spider's Web",
      "author": "David Lagercrantz",
      "coverUrl": "...",
      "category": "Mystery"
    }
  ]
}
```

### Log Reading Session
```
POST /api/mobile/velocity/log
Authorization: Bearer <token>

{
  "userId": "user123",
  "bookId": "book456",
  "pagesRead": 50,
  "durationSeconds": 3600
}
```

### Get Reading Stats
```
GET /api/mobile/velocity/stats/<userId>/<bookId>
Authorization: Bearer <token>
```

## Troubleshooting

### Service Won't Start
1. Check Render logs: `Logs` tab in dashboard
2. Verify data files are committed: `git ls-files | grep -E "\.index|\.pkl|\.csv"`
3. Common error: "FileNotFoundError: books.index not found"
   - Solution: Commit data files to Git and push

### High Memory Usage
- Render Free Tier has 512MB limit
- SentenceTransformer uses ~300MB, FAISS uses ~100MB
- If crashing, consider:
  - Upgrading to Starter Plan ($7/month)
  - Using a smaller embedding model

### Slow Cold Starts
- First request takes 30-60s (model loading)
- Subsequent requests are instant
- Use health endpoint to warm up: `curl https://your-service.onrender.com/api/mobile/health`

## File Structure
```
Python-ranker/
├── mobile_app.py           # Main Flask app
├── reading_velocity.py     # Velocity calculation engine
├── requirements.txt        # Python dependencies
├── runtime.txt            # Python version
├── Procfile               # Render startup command
├── .renderignore          # Files to exclude from build
│
├── books.index            # FAISS index (commit to Git)
├── books_metadata.pkl     # Metadata (commit to Git)
├── book.csv              # Book database (commit to Git)
│
└── .venv/                # Virtual environment (don't commit)
```

## Next Steps
1. ✅ Commit fixed code to GitHub
2. ✅ Deploy to Render
3. ✅ Test health endpoint
4. ✅ Update backend proxy URL in mobile-backend/.env
5. ✅ Test mobile app "AI Suggestions" feature

## Support
For issues, check:
- Render Logs tab
- This README troubleshooting section
- Mobile app error messages
