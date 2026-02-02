import urllib.request
import json
import sys

def check_server(port):
    url = f"http://localhost:{port}/api/tools/lists?length=7&starts_with=A"
    print(f"Checking {url}...")
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            if response.getcode() != 200:
                print(f"Failed with status {response.getcode()}")
                return
            
            data = json.loads(response.read().decode())
            likelihood = data.get('likelihood', [])
            
            if not likelihood:
                print("Likelihood list is empty!")
                return
                
            print(f"\n--- SERVER RESPONSE (Port {port}) ---")
            print(f"Count: {len(likelihood)}")
            print("First 5:", likelihood[:5])
            print("Last 5:", likelihood[-5:])
            
            # Check order of APPOINT and ASPHYXY
            try:
                if 'APPOINT' in likelihood:
                  idx_appoint = likelihood.index('APPOINT')
                else:
                  idx_appoint = -1
                  print("APPOINT missing from list")
                  
                if 'ASPHYXY' in likelihood:
                  idx_asphyxy = likelihood.index('ASPHYXY')
                else: 
                  idx_asphyxy = -1
                  print("ASPHYXY missing from list")

                print(f"\nPOSITIONS:")
                print(f"APPOINT index: {idx_appoint}")
                print(f"ASPHYXY index: {idx_asphyxy}")
                
                if idx_appoint != -1 and idx_asphyxy != -1:
                    if idx_appoint < idx_asphyxy:
                        print("PASS: APPOINT is above ASPHYXY.")
                    else:
                        print("FAIL: APPOINT is below ASPHYXY.")
            except ValueError as e:
                print(f"Error checking indices: {e}")

    except Exception as e:
        print(f"Could not connect to {port}")

check_server(5000)
print("-" * 20)
check_server(3000)
