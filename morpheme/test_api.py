import requests
import json

base_url = "http://localhost:3000"

def test_tournament_status():
    resp = requests.get(f"{base_url}/api/tournament/status")
    print(f"Status Code: {resp.status_code}")
    try:
        data = resp.json()
        print(json.dumps(data, indent=2))
    except Exception as e:
        print(f"Error parsing JSON: {e}")
        print(resp.text)

if __name__ == "__main__":
    test_tournament_status()
