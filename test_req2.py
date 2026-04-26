import requests
import json

payload = {
    "min_rating": 4.5,
    "from_year": 1990,
    "to_year": 2010
}

response = requests.post('http://127.0.0.1:5000/api/recommend/text', json=payload)
print(f"Status Code: {response.status_code}")
print(f"Response: {response.text}")
