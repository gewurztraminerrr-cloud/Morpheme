import sqlite3

conn = sqlite3.connect("/Users/jeffbabiak/.gemini/antigravity/scratch/morpheme/morpheme.db")
conn.row_factory = sqlite3.Row

user = conn.execute("SELECT * FROM users WHERE username = 'jeffy' COLLATE NOCASE").fetchone()
if user:
    print("=== USERS TABLE ===")
    for k in user.keys():
        print(f"{k}: {user[k]}")
else:
    print("User jeffy not found in users table.")

ratings = conn.execute("SELECT * FROM user_ratings WHERE user_id = (SELECT id FROM users WHERE username = 'jeffy' COLLATE NOCASE)").fetchall()
print("\n=== USER_RATINGS TABLE ===")
for r in ratings:
    print(dict(r))

conn.close()
