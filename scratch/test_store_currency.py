import os
import json
import urllib.parse

def test_catalog():
    catalog_path = "static/data/store_catalog.json"
    assert os.path.exists(catalog_path), f"Catalog not found at {catalog_path}"
    with open(catalog_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "last_catalog_update" in data, "Missing last_catalog_update"
    assert "products" in data, "Missing products list"
    assert len(data["products"]) >= 17, f"Expected at least 17 products, found {len(data['products'])}"

    categories = set()
    for p in data["products"]:
        assert "id" in p, f"Product missing id: {p}"
        assert "title" in p, f"Product missing title: {p}"
        assert "base_price_usd" in p, f"Product missing base_price_usd: {p}"
        assert "last_changed_date" in p, f"Product missing last_changed_date: {p}"
        assert "image_url" in p, f"Product missing image_url: {p}"
        assert "buy_url" in p, f"Product missing buy_url: {p}"
        categories.add(p["category"])

        # Check image exists on disk
        img_rel = p["image_url"].split("?")[0].lstrip("/")
        assert os.path.exists(img_rel), f"Image does not exist on disk: {img_rel}"

    print(f"Catalog Verified: {len(data['products'])} products across categories: {categories}")
    print("All image files exist on disk!")

if __name__ == "__main__":
    test_catalog()
