import sqlite3
conn = sqlite3.connect('mappings.db')
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = cursor.fetchall()
print('Tables:', tables)
for t in tables:
    cursor.execute(f"PRAGMA table_info({t[0]})")
    print(f"\nSchema of {t[0]}:", cursor.fetchall())
    cursor.execute(f"SELECT * FROM {t[0]} LIMIT 2")
    print(f"Sample data:", cursor.fetchall())
conn.close()