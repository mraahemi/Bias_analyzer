from flask import Flask, render_template, url_for, request, redirect
from flask_sqlalchemy import SQLAlchemy
import sqlite3
from datetime import datetime

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///test.db'

@app.route('/', methods=['POST', 'GET'])
def index():
    conn = get_db_connection()
    models = conn.execute('SELECT * FROM models').fetchall()
    conn.close()
    # todo: add possibility to upload model
    return render_template('index.html', models=models)

@app.route('/models/<int:id>')
def show_model(id):
    # todo: can we make this a global function somehow???
    conn = get_db_connection()
    # todo: extract only one line here so that we don't need to loop in html
    model_overview = conn.execute(f'SELECT * FROM models where id={id}').fetchall()
    return render_template('model_overview.html', model_overview=model_overview)

def get_db_connection():
    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row
    return conn

if __name__ == "__main__":
    app.run(debug=True, use_debugger=False, use_reloader=False)
