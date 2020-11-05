from flask import Flask, render_template, url_for, request, redirect
from flask_sqlalchemy import SQLAlchemy
import sqlite3
from datetime import datetime

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///test.db'
db = SQLAlchemy(app)

class Todo(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.String(200), nullable=False)
    date_created = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return '<Task %r>' % self.id

@app.route('/', methods=['POST', 'GET'])
def index():
    conn = get_db_connection()
    models = conn.execute('SELECT * FROM models').fetchall()
    conn.close()
    # todo: add possibility to upload model
    return render_template('index.html', models=models)

@app.route('/models/<int:id>')
def show_model(id):
    conn = get_db_connection()
    model_overview = conn.execute(f'SELECT * FROM models where id={id}').fetchall()
    print('this is the model overview')
    print(model_overview)
    return render_template('model_overview.html', model_overview=model_overview)

def get_db_connection():
    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row
    return conn

if __name__ == "__main__":
    app.run(debug=True, use_debugger=False, use_reloader=False)
