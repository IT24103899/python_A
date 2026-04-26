import requests
import json

url = "http://localhost:5000/api/recommend/personal"

print("--- Testing Discovery Mode (Empty History) ---")
try:
    response = requests.post(url, json={}, timeout=10)
    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"Type: {data.get('type')}")
    print(f"Count: {len(data.get('recommendations', []))}")
    if data.get('recommendations'):
        print(f"First Book: {data['recommendations'][0]['title']}")
except Exception as e:
    print(f"Error: {e}")

print("\n--- Testing Personalized Mode (Mock History) ---")
try:
    # Assuming IDs like 1, 2, 3 exist in the 5000+ book dataset
    response = requests.post(url, json={"bookIds": [1, 2, 3]}, timeout=10)
    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"Type: {data.get('type')}")
    print(f"Count: {len(data.get('recommendations', []))}")
except Exception as e:
    print(f"Error: {e}")
