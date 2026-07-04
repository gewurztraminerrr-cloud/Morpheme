import pexpect
import sys

def main():
    password = "CT4n2S#sQ918"
    ip = "132.148.72.249"
    
    print(f"Connecting to {ip} via SSH...")
    child = pexpect.spawn(f'ssh morpheme@{ip}', encoding='utf-8', timeout=20)
    child.logfile_read = sys.stdout
    
    child.expect([r"password:"])
    child.sendline(password)
    
    child.expect([r"\$", r"#"])
    print("\nLogged in successfully!")
    
    db_path = "/home/morpheme/morpheme/morpheme.db"
    
    # Python script to analyze human stats by rating bracket
    script_content = """
import sqlite3
import json

conn = sqlite3.connect('/home/morpheme/morpheme/morpheme.db')
cur = conn.cursor()

# Query: Get stats for all rounds played by registered human users (user_id > 0)
cur.execute('''
    SELECT user_rating, wpm, total_score, words_json, board_dimensions, round_duration
    FROM round_history
    WHERE user_id > 0 AND round_duration < 7200 AND total_score > 0
''')
rows = cur.fetchall()

brackets = {
    '400-999': [],
    '1000-1199': [],
    '1200-1399': [],
    '1400-1599': [],
    '1600-1799': [],
    '1800-1999': [],
    '2000+': []
}

for r in rows:
    rating, wpm, score, wjson, dims, dur = r
    if rating < 1000: b = '400-999'
    elif rating < 1200: b = '1000-1199'
    elif rating < 1400: b = '1200-1399'
    elif rating < 1600: b = '1400-1599'
    elif rating < 1800: b = '1600-1799'
    elif rating < 2000: b = '1800-1999'
    else: b = '2000+'
    
    try:
        words = json.loads(wjson)
        num_words = len(words)
        avg_len = sum(len(w['word']) for w in words) / num_words if num_words > 0 else 0
    except:
        num_words = 0
        avg_len = 0
        
    brackets[b].append({
        'wpm': wpm or 0,
        'score': score,
        'num_words': num_words,
        'avg_len': avg_len,
        'duration': dur
    })

print('--- HUMAN STATS BY RATING BRACKET ---')
for b, data in brackets.items():
    if not data:
        print(f"Bracket {b}: No data")
        continue
    avg_wpm = sum(d['wpm'] for d in data) / len(data)
    avg_score = sum(d['score'] for d in data) / len(data)
    avg_words = sum(d['num_words'] for d in data) / len(data)
    avg_len = sum(d['avg_len'] for d in data) / len(data)
    print(f"Bracket {b} ({len(data)} rounds):")
    print(f"  Avg WPM: {avg_wpm:.2f}")
    print(f"  Avg Words Found: {avg_words:.2f}")
    print(f"  Avg Score: {avg_score:.2f}")
    print(f"  Avg Word Length: {avg_len:.2f}")
"""
    
    print("\n--- Writing stats script to remote /tmp/query_human_stats.py ---")
    child.sendline("cat << 'EOF' > /tmp/query_human_stats.py")
    child.sendline(script_content.strip())
    child.sendline("EOF")
    child.expect([r"\$", r"#"])
    
    print("\n--- Running stats script ---")
    child.sendline("python3 /tmp/query_human_stats.py")
    child.expect([r"\$", r"#"])
    
    print("\n--- Cleaning up ---")
    child.sendline("rm /tmp/query_human_stats.py")
    child.expect([r"\$", r"#"])
    
    child.sendline("exit")
    child.close()

if __name__ == "__main__":
    main()
