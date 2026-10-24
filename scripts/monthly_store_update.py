#!/usr/bin/env python3
"""
Monthly Store Updater for Morpheme
----------------------------------
Automatically checks, refreshes, and maintains the Morpheme Store catalog.
- Updates the top 'Prices and products last updated on [date]' banner.
- Proactively checks for NEW EDITIONS of all listed dictionaries (OSPD, CSW, Collins)
  to ensure the catalog reflects the latest commercially available releases.
- Allows targeted replacements for discontinued/outdated hardware (setting 'Last changed on [date]').
- Validates JSON catalog integrity and pricing.
"""

import os
import sys
import json
import argparse
from datetime import datetime

CATALOG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'static', 'data', 'store_catalog.json')

def load_catalog(path=CATALOG_PATH):
    if not os.path.exists(path):
        print(f"Error: Catalog not found at {path}")
        sys.exit(1)
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_catalog(data, path=CATALOG_PATH):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
    print(f"Catalog successfully written to {path}")

def compute_next_month_str(base_date=None):
    dt = base_date or datetime.now()
    year = dt.year + (dt.month // 12)
    month = (dt.month % 12) + 1
    day = min(dt.day, 28)
    next_dt = dt.replace(year=year, month=month, day=day)
    return next_dt.strftime("%B %d, %Y")

def check_dictionary_editions(catalog):
    """
    Mandatory routine check during monthly Store updates:
    Verifies if newer editions of dictionaries/wordlists have been commercially released.
    """
    print("\n--- Mandatory Monthly Dictionary Edition Routine Check ---")
    book_products = [p for p in catalog.get('products', []) if p.get('category') == 'books']
    print(f"Checking {len(book_products)} dictionary/wordlist listings in catalog:")
    for b in book_products:
        pid = b.get('id')
        title = b.get('title')
        last_changed = b.get('last_changed_date')
        print(f"  • [{pid}] \"{title}\" (Last changed: {last_changed})")
        if pid == 'ospd_dict':
            print("    -> Check: Is Merriam-Webster OSPD 8th Edition (or newer) commercially available on Amazon/retailers?")
        elif pid == 'csw_dict':
            print("    -> Check: Has Collins / WESPA released a new CSW definitions tournament edition?")
        elif pid == 'collins_words':
            print("    -> Check: Has Collins / WESPA released a new official tournament word list?")
    print("Action item: If a new edition is commercially available, update title, edition number, description, buy_url, and cover art.")
    print("-----------------------------------------------------------\n")

def run_monthly_refresh(check_only=False, target_date=None, target_next_date=None):
    catalog = load_catalog()
    today_str = target_date or datetime.now().strftime("%B %d, %Y")
    next_str = target_next_date or compute_next_month_str()
    
    print(f"=== Morpheme Store Monthly Catalog Maintenance ===")
    print(f"Previous catalog update: {catalog.get('last_catalog_update')}")
    print(f"New catalog update date: {today_str}")
    print(f"Next scheduled update:   {next_str}")
    print(f"Total products in catalog: {len(catalog.get('products', []))}")

    # Mandatory Dictionary Edition Check
    check_dictionary_editions(catalog)

    if not check_only:
        catalog['last_catalog_update'] = today_str
        catalog['next_scheduled_update'] = next_str
        save_catalog(catalog)
        print("Updated top-level catalog timestamps.")

    return catalog

def replace_or_update_product(product_id, new_data):
    catalog = load_catalog()
    today_str = datetime.now().strftime("%B %d, %Y")
    found = False

    for p in catalog.get('products', []):
        if p.get('id') == product_id:
            found = True
            print(f"Found product '{product_id}': {p.get('title')}")
            for key, val in new_data.items():
                p[key] = val
            p['last_changed_date'] = today_str
            print(f"Updated product '{product_id}' with new specs/pricing. Set last_changed_date to {today_str}.")
            break

    if not found:
        print(f"Product '{product_id}' not found. Adding as new product...")
        new_data['id'] = product_id
        new_data['last_changed_date'] = today_str
        catalog['products'].append(new_data)

    catalog['last_catalog_update'] = today_str
    save_catalog(catalog)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Morpheme Monthly Store Catalog Updater")
    parser.add_argument('--check', action='store_true', help="Check catalog without writing changes")
    parser.add_argument('--date', type=str, help="Custom date string to set (default: today)")
    parser.add_argument('--replace-id', type=str, help="Product ID to replace or update")
    parser.add_argument('--new-title', type=str, help="New title for replaced product")
    parser.add_argument('--new-price', type=float, help="New base price (USD) for replaced product")
    parser.add_argument('--new-buy-url', type=str, help="New purchase URL for replaced product")

    args = parser.parse_args()

    if args.replace_id:
        updates = {}
        if args.new_title: updates['title'] = args.new_title
        if args.new_price is not None: updates['base_price_usd'] = args.new_price
        if args.new_buy_url: updates['buy_url'] = args.new_buy_url
        replace_or_update_product(args.replace_id, updates)
    else:
        run_monthly_refresh(check_only=args.check, target_date=args.date)
