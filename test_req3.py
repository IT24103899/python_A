import requests
import json

payload = {
    "author": "George Orwell",
    "min_rating": 3.0,
    "from_year": 2010,
    "to_year": 2019
}

response = requests.post('http://127.0.0.1:5000/api/recommend/text', json=payload)
print(f"Status Code: {response.status_code}")
try:
    data = response.json()
    print(f"Recommendations: {len(data.get('recommendations', []))}")
    print(data)
except Exception as e:
    print(f"Error parsing JSON: {e}")
    print(response.text)
