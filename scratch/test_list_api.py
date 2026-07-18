import app
import time
from app import app as flask_app

def main():
    print("Testing tools_get_lists api call locally...")
    from word_validator import word_validator
    t_csw = time.time()
    word_validator.ensure_csw_loaded()
    print(f"CSW Load took {time.time() - t_csw:.4f}s")
    
    with flask_app.test_request_context('/api/tools/lists?list_type=nwl&length=all&starts_with=all'):
        t0 = time.time()
        res = app.tools_get_lists()
        print(f"Status code: {res.status_code}")
        print(f"Time taken: {time.time() - t0:.4f}s")
        data = res.get_json()
        print(f"Number of NWL words returned: {len(data.get('nwl', []))}")

if __name__ == "__main__":
    main()
