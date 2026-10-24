from app import app

client = app.test_client()
res = client.get('/api/store/catalog')
print("Status Code:", res.status_code)
assert res.status_code == 200
data = res.get_json()
print("Catalog Top Date:", data.get('last_catalog_update'))
print("Products Count:", len(data.get('products', [])))
print("Cache Control Header:", res.headers.get('Cache-Control'))
assert len(data.get('products', [])) == 17
print("Flask endpoint test passed successfully!")
