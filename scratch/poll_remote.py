import urllib.request
import time
import json

def main():
    url = "https://morpheme.games/api/lobby-stats"
    print(f"Polling {url} for 60 seconds...")
    start_time = time.time()
    
    while time.time() - start_time < 60:
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'MorphemeApp-Diagnostic'})
            with urllib.request.urlopen(req, timeout=3) as response:
                status = response.status
                body = response.read().decode('utf-8')
                print(f"[{time.strftime('%H:%M:%S')}] Status: {status} | Body: {body}")
        except urllib.error.HTTPError as e:
            print(f"[{time.strftime('%H:%M:%S')}] HTTP Error: {e.code}")
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] Connection Error: {e}")
        time.sleep(1)

if __name__ == "__main__":
    main()
