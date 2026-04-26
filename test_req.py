import requests
import json

payload = {
    "query": "magic school",
    "min_rating": 4.5,
    "from_year": 1990,
    "to_year": 2010
}

response = requests.post('http://127.0.0.1:5000/api/recommend/text', json=payload)
data = response.json()

if data.get('recommendations'):
    for rec in data['recommendations']:
        print(f"[{rec['match_score']}%] {rec['title']} by {rec['author']}")
else:
    print("No recommendations found or backend not running")
