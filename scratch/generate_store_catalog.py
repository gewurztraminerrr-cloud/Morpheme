import json

catalog = {
    "last_catalog_update": "October 16, 2026",
    "products": [
        {
            "id": "stylus",
            "category": "accessories",
            "title": "Yacig 4-in-1 Capacitive Stylus",
            "base_price_usd": 12.99,
            "last_changed_date": "October 16, 2026",
            "image_url": "/static/images/stylus.png?v=33139",
            "desc": [
                "High-precision 4-in-1 stylus with clear disc tip, mesh fiber, and black rubber tips. Perfect for Morpheme mobile and tablet play!"
            ],
            "features": [
                "4-in-1 Tip Design",
                "Universal Compatibility",
                "No Batteries Required",
                "Free Replacement Tips"
            ],
            "buy_url": "https://www.amazon.ca/Yacig-Capacitive-Sensitivity-Precision-Compatible/dp/B07GKY9FJZ?th=1",
            "buy_label": "View on Amazon"
        },
        {
            "id": "mousepad",
            "category": "accessories",
            "title": "Ergonomic Mousepad with Wrist Support",
            "base_price_usd": 9.49,
            "last_changed_date": "October 16, 2026",
            "image_url": "/static/images/mousepad.png?v=33139",
            "desc": [
                "High-quality non-slip rubber surface with an integrated, raised wrist rest. This is perfect for repurposing as an ergonomic support while playing Morpheme with a stylus, reducing strain during high-speed word searching."
            ],
            "features": [
                "Built-in Ergonomic Wrist Support",
                "Non-Slip Rubber Base",
                "Smooth Microfiber Surface",
                "Portable & Lightweight"
            ],
            "buy_url": "https://www.amazon.ca/Non-Slip-Rubber-Computer-Laptop-Office/dp/B006ZAUD10",
            "buy_label": "View on Amazon"
        },
        {
            "id": "csw_dict",
            "category": "books",
            "title": "Official Scrabble Dictionary (CSW)",
            "base_price_usd": 17.99,
            "last_changed_date": "October 16, 2026",
            "image_url": "/static/images/csw_dict_29.jpg?v=1",
            "desc": [
                "The official, authoritative dictionary for Scrabble and international word tournaments (CSW), containing 2–9 letter words and their definitions. Essential for mastering all valid playable words and understanding their meanings in Morpheme's CSW dictionary mode."
            ],
            "features": [
                "Official CSW Dictionary",
                "Includes 2–9 Letter Words & Definitions",
                "International Tournament Standard",
                "Complete Playable Words & Endings"
            ],
            "buy_url": "https://www.amazon.ca/SCRABBLETM-Dictionary-official-bestselling-playable/dp/0008660735",
            "buy_label": "View on Amazon"
        },
        {
            "id": "ospd_dict",
            "category": "books",
            "title": "The Official Scrabble Players Dictionary (OSPD)",
            "base_price_usd": 14.50,
            "last_changed_date": "October 16, 2026",
            "image_url": "/static/images/scrabble_players_dict_7.png?v=33141",
            "desc": [
                "The official North American Scrabble Players Dictionary (OSPD, Seventh Edition) published by Merriam-Webster, containing words and their definitions 2–8 letters long. This dictionary is not identical to NWL, but close enough to render it worth having for studying and reference."
            ],
            "features": [
                "Official Merriam-Webster OSPD (Seventh Edition)",
                "Words & Definitions 2–8 Letters Long",
                "Close match to NWL standard",
                "Essential Reference for Word Study"
            ],
            "buy_url": "https://www.amazon.ca/dp/0877795959",
            "buy_label": "View on Amazon"
        },
        {
            "id": "collins_words",
            "category": "books",
            "title": "Official SCRABBLE Words (Collins 2–15 All Words)",
            "base_price_usd": 21.99,
            "last_changed_date": "October 16, 2026",
            "image_url": "/static/images/collins_words_green.png?v=33140",
            "desc": [
                "The complete, unabridged official tournament wordlist published by Collins and endorsed by WESPA, containing over 276,000 playable words from 2 to 15 letters with inflected forms. The first half covers all 2-to-9 letter words, and the other half covers all 10-to-15 letter words. (Note: This book is strictly a word list — there are no definitions.)"
            ],
            "features": [
                "Complete 2-to-15 Letter Word List",
                "Word List Only (No Definitions)",
                "2–9 Letter Words in First Half",
                "10–15 Letter Words in Second Half",
                "Endorsed by WESPA & International Tournament Standard"
            ],
            "buy_url": "https://www.amazon.ca/Collins-Official-Scrabble-Words-aa/dp/0007537972",
            "buy_label": "View on Amazon"
        },
        {
            "id": "boggle_classic",
            "category": "games",
            "title": "Boggle Classic (4×4) — The Original Word Rush",
            "base_price_usd": 14.99,
            "last_changed_date": "October 16, 2026",
            "image_url": "/static/images/boggle_classic.jpg?v=33139",
            "desc": [
                "The game that Morpheme was born from! This original 4×4 grid gives you 16 lettered cubes and a 3-minute timer to find as many words as possible — identical in spirit to a Morpheme 4×4 blitz round. It is the single best physical training tool for building spatial word-pattern recognition.",
                "Playing Classic Boggle offline trains your eye to scan letter adjacencies at speed — the exact same skill used in Morpheme's path-finding."
            ],
            "features": [
                "16 Lettered Cubes (4×4)",
                "3-Minute Sand Timer",
                "Scoring Pad Included",
                "2+ Players"
            ],
            "buy_url": "https://www.amazon.ca/Hasbro-HASC2187-Boggle-Classic-Game/dp/B073SMSKZH",
            "buy_label": "View on Amazon"
        },
        {
            "id": "boggle_big",
            "category": "games",
            "title": "Big Boggle (5×5) — Extended Grid Challenge",
            "base_price_usd": 21.99,
            "last_changed_date": "October 16, 2026",
            "image_url": "/static/images/boggle_big.jpg?v=33139",
            "desc": [
                "Big Boggle expands the grid to 5×5 — 25 cubes, a 4-letter minimum word requirement, and exponentially more possible word paths. This directly mirrors Morpheme's 5×5 board mode.",
                "The 5×5 grid forces you to think in longer word chains, training the deeper vocabulary and path-finding instincts required for Morpheme's 5×5 boards where longer words score disproportionately higher."
            ],
            "features": [
                "25 Lettered Cubes (5×5)",
                "4-Letter Minimum Required",
                "3-Minute Sand Timer",
                "Deeper Vocabulary Training"
            ],
            "buy_url": "https://www.amazon.ca/Winning-Moves-1147-Big-Boggle/dp/1223063119",
            "buy_label": "View on Amazon"
        },
        {
            "id": "boggle_super_big",
            "category": "games",
            "title": "Super Big Boggle (6×6) — Tournament Grade",
            "base_price_usd": 28.99,
            "last_changed_date": "October 16, 2026",
            "image_url": "/static/images/boggle_super_big.jpg?v=33139",
            "desc": [
                "Super Big Boggle takes the grid all the way to 6×6 — 36 cubes including special double-letter and blank cubes. This is the closest physical equivalent to Morpheme's flagship 6×8 extended boards.",
                "Mastering the 6×6 grid forces you to navigate complex multi-directional adjacency chains — exactly the advanced skill needed to dominate Morpheme's largest boards."
            ],
            "features": [
                "36 Cubes incl. Blanks (6×6)",
                "Double-Letter Special Cube",
                "Highest Complexity Available",
                "Tournament-Level Training"
            ],
            "buy_url": "https://www.amazon.ca/Winning-Moves-1165-Super-Boggle/dp/B008GCT8ZM/",
            "buy_label": "View on Amazon"
        },
        {
            "id": "joycat_letters",
            "category": "accessories",
            "title": "JoyCat Silicone Uppercase Magnetic Letters (26 Pcs)",
            "base_price_usd": 14.49,
            "last_changed_date": "October 16, 2026",
            "image_url": "/static/images/joycat_letters.png?v=33139",
            "desc": [
                "Active Word Recall: Spell out challenging words or stems you struggle with so you are constantly reminded of them every time you open the fridge!",
                "Soft, Durable & Safe: Made from high-quality, non-toxic food-grade silicone with strong embedded magnets. Perfect for refrigerators, whiteboards, or metal study desks."
            ],
            "features": [
                "26 Silicone Uppercase Letters (A–Z)",
                "Strong Embedded Fridge Magnets",
                "Durable, Washable Food-Grade Silicone",
                "Includes Cotton Storage Bag"
            ],
            "buy_url": "https://www.amazon.ca/JoyCat-Silicone-Uppercase-Refrigerator-Preschool/dp/B0F8MQQSXM",
            "buy_label": "View on Amazon"
        },
        {
            "id": "samsung_s25_ultra",
            "category": "mobile",
            "title": "Samsung Galaxy S25 Ultra (The Ultimate Mobile Platform)",
            "base_price_usd": 1299.99,
            "last_changed_date": "October 16, 2026",
            "image_url": "/static/images/phones25_final.png?v=33139",
            "desc": [
                "The Perfect Choice: With an expansive 6.9\" Dynamic AMOLED 2X display and an integrated S-Pen, this is the ultimate way to play Morpheme on mobile. The S-Pen delivers surgical, zero-drag precision for complex word paths across large boards.",
                "Motion Sensitivity: Features a high-fidelity 120Hz refresh rate and ultra-responsive touch sampling, ensuring every swipe tracks with zero perceptible latency during tournament blitz rounds.",
                "Performance: Powered by the Snapdragon 8 Elite and 12GB of RAM, this powerhouse handles 6x8 boards and real-time animations with total fluid stability."
            ],
            "features": [
                "6.9\" 120Hz Dynamic AMOLED 2X",
                "Integrated Precision S-Pen",
                "Snapdragon 8 Elite Processor",
                "Titanium Armor Frame & Anti-Reflective Glass"
            ],
            "buy_url": "https://www.amazon.ca/Samsung-Galaxy-AMOLED-Display-AI-Enhanced/dp/B0F2ZS5VC4",
            "buy_label": "View on Amazon"
        },
        {
            "id": "oneplus_15",
            "category": "mobile",
            "title": "OnePlus 15 (Extreme Speed Platform)",
            "base_price_usd": 999.99,
            "last_changed_date": "October 16, 2026",
            "image_url": "/static/images/oneplus15_final.png?v=33139",
            "desc": [
                "The Record Breaker: If you value pure speed, the OnePlus 15 is built for you. Its 6.78\" AMOLED screen supports an extraordinary 165Hz Refresh Rate, offering the smoothest Morpheme experience possible.",
                "Zero-Latency Swiping: With its class-leading 165Hz panel, the latency between your finger and the word-pathing engine is decimated, allowing for insane words-per-minute (WPM) rates.",
                "Performance: Packed with the Snapdragon 8 Elite and 16GB RAM, it provides uncompromising sustained power that never throttles during intense tournament sessions."
            ],
            "features": [
                "6.78\" 165Hz Ultra-Fluid AMOLED Display",
                "Snapdragon 8 Elite Processor",
                "16GB High-Speed LPDDR5X RAM",
                "Dual-Vapor Chamber Liquid Cooling"
            ],
            "buy_url": "https://www.bestbuy.ca/en-ca/product/oneplus-15-512gb-infinite-black-unlocked/19634597",
            "buy_label": "View on Best Buy"
        },
        {
            "id": "iphone_17_pro",
            "category": "mobile",
            "title": "Apple iPhone 17 Pro (Elite Mobile Gaming Platform)",
            "base_price_usd": 1199.00,
            "last_changed_date": "October 16, 2026",
            "image_url": "/static/images/iphone17_final.png?v=33139",
            "desc": [
                "Why It's Perfect: The iPhone 17 Pro features the A19 Pro chip with laser-welded vapor chamber cooling, ensuring your Morpheme sessions stay locked at 120FPS even during the most complex 6x8 grid animations.",
                "Touch & Motion Sensitivity: The 120Hz ProMotion display offers the lowest touch latency in its class, perfectly syncing each swipe with the game's pathing engine.",
                "Tactile Feedback: Advanced Haptics through the Taptic Engine provide crisp, tactile confirmation for every letter you select, significantly reducing errors during high-speed competitive play."
            ],
            "features": [
                "Super Retina XDR with 120Hz ProMotion",
                "Apple A19 Pro Flagship Chip",
                "Advanced Ceramic Shield Generation 2",
                "Ultra-Crisp Taptic Letter Confirmation"
            ],
            "buy_url": "https://www.bestbuy.ca/en-ca/product/open-box-apple-iphone-17-pro-512gb-deep-blue-unlocked/19483947",
            "buy_label": "View on Best Buy"
        },
        {
            "id": "ipad_pro_11_m5",
            "category": "tablet",
            "title": "Apple iPad Pro 11-inch (M5 Pro Platform)",
            "base_price_usd": 999.00,
            "last_changed_date": "October 16, 2026",
            "image_url": "/static/images/ipadpro11m5_final.png?v=33139",
            "desc": [
                "The Ultimate Canvas: Featuring Apple's blazing M5 chip, this is the most powerful handheld Morpheme device ever made. Its Ultra Retina XDR (Tandem OLED) display provides extreme contrast and vibrancy across all 39 Morpheme color themes.",
                "High Precision: Optimized for high-sensitivity Apple Pencil Pro use, including hover detection that allows you to see your word path before you even touch the screen. The 120Hz ProMotion ensures instant response.",
                "Tournament Form Factor: The ultra-thin 11-inch design is the Morpheme community's favorite—offering a massive, clear view of the board while remaining light enough for comfortable hand-held tournament use."
            ],
            "features": [
                "11\" Ultra Retina XDR (Tandem OLED) Display",
                "Apple Silicon M5 Architecture",
                "Apple Pencil Pro Hover & Squeeze Support",
                "120Hz ProMotion Ultra-Low Latency"
            ],
            "buy_url": "https://www.amazon.ca/Apple-iPad-Pro-11-inch-M5/dp/B0FWDD257H?th=1",
            "buy_label": "View on Amazon"
        },
        {
            "id": "asus_zephyrus_g16",
            "category": "pc_laptop",
            "title": "ASUS ROG Zephyrus G16 (Tournament PC Laptop)",
            "base_price_usd": 1999.99,
            "last_changed_date": "October 16, 2026",
            "image_url": "/static/images/asus_zephyrus_g16.jpg?v=1",
            "desc": [
                "The Ultimate Competitive PC Laptop: Equipped with a stunning 16-inch 240Hz ROG Nebula OLED display (0.2ms response time), the Zephyrus G16 completely eliminates motion blur during fast letter scanning on 6x8 and 3D boards.",
                "Tournament-Grade Keyboard: Featuring stealth-action mechanical actuation switches with 1.7mm key travel and full N-key rollover, every keystroke inputs instantaneously without key chattering or dropped inputs during blitz rounds.",
                "Whisper-Quiet Performance: Advanced vapor chamber liquid metal cooling keeps fan noise virtually silent, allowing players to focus with total concentration during championship matches."
            ],
            "features": [
                "16\" 2.5K 240Hz 0.2ms ROG Nebula OLED Display",
                "Intel Core Ultra 9 / RTX Flagship Architecture",
                "N-Key Rollover Tournament-Grade Keyboard",
                "CNC Aluminum Unibody Chassis Under 4 lbs"
            ],
            "buy_url": "https://www.bestbuy.ca/en-ca/product/asus-rog-zephyrus-g16-16-oled-240hz-gaming-laptop-eclipse-grey-intel-core-ultra-9-1tb-ssd-32gb-ram-rtx-4080/17698889",
            "buy_label": "View on Best Buy"
        },
        {
            "id": "morpheme_pc_desktop",
            "category": "pc_desktop",
            "title": "Morpheme Pro Tournament PC Desktop Rig",
            "base_price_usd": 2199.00,
            "last_changed_date": "October 16, 2026",
            "image_url": "/static/images/pc_desktop_rig.jpg?v=1",
            "desc": [
                "Peak Esports Performance: Designed for serious competitive Morpheme players who stream, practice multi-board analysis, and compete in live brackets. Boasts top-tier single-core CPU speeds for instant JavaScript DOM rendering and sub-millisecond WebSocket validation.",
                "Zero Input Latency: Fine-tuned BIOS and DPC latency architecture ensures that high-polling-rate gaming keyboards and mice register word submissions without a single frame of lag.",
                "Multi-Display Command Center: Easily powers dual 240Hz+ gaming monitors so you can keep live tournament brackets, dictionary lookups, and Morpheme active simultaneously."
            ],
            "features": [
                "Ultra-High Single-Core Frequency Gaming CPU",
                "32GB High-Speed DDR5 RAM & Gen4 NVMe SSD",
                "Custom Liquid Cooling for Absolute Silence",
                "Multi-Monitor 240Hz+ DisplayPort Outputs"
            ],
            "buy_url": "https://www.bestbuy.ca/en-ca/collection/gaming-desktop-computers/26262",
            "buy_label": "View on Best Buy"
        },
        {
            "id": "macbook_pro_16_m4",
            "category": "mac_laptop",
            "title": "Apple 16-inch MacBook Pro (M4 Pro / Max Flagship)",
            "base_price_usd": 2499.00,
            "last_changed_date": "October 16, 2026",
            "image_url": "/static/images/macbook_pro_16.jpg?v=1",
            "desc": [
                "The Elite Mac Laptop: Built around the M4 Pro chip, the 16-inch MacBook Pro delivers blistering single-core performance that makes Morpheme's real-time word lookup and animation run with locked 120FPS fluidity.",
                "Liquid Retina XDR Display: With up to 1,600 nits peak brightness and 120Hz ProMotion, contrast across dark and light Morpheme themes is breathtaking. Every letter tile stands out with razor-sharp distinction.",
                "Best-in-Class Magic Keyboard: Scissor-switch keys offer tactile, satisfying key travel with zero debounce latency, ensuring word-after-word speed typing with supreme comfort during long tournament sessions."
            ],
            "features": [
                "16.2\" Liquid Retina XDR with 120Hz ProMotion",
                "Apple M4 Pro 14-Core CPU / 20-Core GPU",
                "Precision Magic Keyboard with Full-Height Function Row",
                "Over 22 Hours of Battery Life for All-Day Tournament Play"
            ],
            "buy_url": "https://www.apple.com/shop/buy-mac/macbook-pro/16-inch",
            "buy_label": "View on Apple"
        },
        {
            "id": "mac_studio_display",
            "category": "mac_desktop",
            "title": "Apple Mac Studio & 5K Studio Display (Master Analyzer Rig)",
            "base_price_usd": 3598.00,
            "last_changed_date": "October 16, 2026",
            "image_url": "/static/images/mac_studio_display.jpg?v=1",
            "desc": [
                "The Ultimate Analytical Setup: For players who study word lists, test high-depth Combo Checker algorithms, and run tournament broadcasts. The Mac Studio packs immense CPU bandwidth in an ultra-compact desktop footprint.",
                "Pixel-Perfect 5K Studio Display: Running Morpheme on a native 5K resolution canvas provides unprecedented board legibility with zero scaling artifacts. See every tile, path, and definition in immaculate detail.",
                "Whisper-Silent Thermal Architecture: Even under heavy background analysis or high-speed multi-tab word queries, the Mac Studio remains completely silent."
            ],
            "features": [
                "Apple M4 Max / M3 Ultra Desktop Chipset",
                "Paired with 27-inch 5K Retina Studio Display (600 nits)",
                "Support for up to Five External Displays",
                "Ultra-Low Latency Thunderbolt 4 / USB-C Ports"
            ],
            "buy_url": "https://www.apple.com/shop/buy-mac/mac-studio",
            "buy_label": "View on Apple"
        }
    ]
}

with open("static/data/store_catalog.json", "w") as f:
    json.dump(catalog, f, indent=4)

print(f"Successfully generated store_catalog.json with {len(catalog['products'])} products!")
