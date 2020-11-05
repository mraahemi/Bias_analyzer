import sqlite3

connection = sqlite3.connect('database.db')


with open('schema.sql') as f:
    connection.executescript(f.read())

cur = connection.cursor()

cur.execute("INSERT INTO models (id, bias, content) VALUES (?, ?, ?)",
            ('1', 'heavily biased', 'Scoring Model 1')
            )

cur.execute("INSERT INTO models (id, bias, content) VALUES (?, ?, ?)",
            ('2', 'heavily biased', 'Scoring Model 2')
            )

cur.execute("INSERT INTO models (id, bias, content) VALUES (?, ?, ?)",
            ('3', 'not biased', 'Scoring Model 3')
            )

connection.commit()
connection.close()
