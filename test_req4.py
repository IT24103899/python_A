import requests
import json

payload = {
    "author": "George Orwell",
    "min_rating": 3.0
}

response = requests.post('http://127.0.0.1:5000/api/recommend/text', json=payload)
data = response.json()
print(f"Recommendations: {len(data.get('recommendations', []))}")
if data.get('recommendations'):
    print(f"First: {data['recommendations'][0]['title']} ({data['recommendations'][0]['original_publication_year']})")
