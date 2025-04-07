import sqlite3
conn = sqlite3.connect('clients.db')
cursor = conn.cursor()

cursor.execute("SELECT * FROM client")
clients = cursor.fetchall()
for client in clients:
    print(client)

conn.close()