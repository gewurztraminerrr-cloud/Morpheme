import urllib.request
import json

def main():
    url = "http://132.148.72.249/api/tools/lists?list_type=added"
    req = urllib.request.Request(url, headers={'Host': 'morpheme.games'})
    try:
        print("Querying remote lists API...")
        with urllib.request.urlopen(req, timeout=15) as response:
            data = json.loads(response.read().decode())
            added_words = data.get('added', [])
            print(f"API returned {len(added_words)} Added Words.")
            # Check if any returned word is in CSW
            # (let's check a sample of words)
            print(f"First 10 words: {added_words[:10]}")
    except Exception as e:
        print(f"Error querying API: {e}")

if __name__ == "__main__":
    main()
