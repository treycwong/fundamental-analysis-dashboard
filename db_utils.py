# db_utils.py
import sqlite3
from datetime import datetime

def init_db():
    conn = sqlite3.connect('gold_analysis.db')
    c = conn.cursor()

    # Create tables if they don't exist
    c.execute('''
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY,
            date TEXT,
            title TEXT,
            category TEXT,
            impact INTEGER,
            notes TEXT,
            completed INTEGER DEFAULT 0
        )
    ''')

    c.execute('''
        CREATE TABLE IF NOT EXISTS scores (
            id INTEGER PRIMARY KEY,
            date TEXT,
            interest_rates REAL,
            inflation REAL,
            dollar_strength REAL,
            supply REAL,
            demand REAL,
            positioning REAL,
            notes TEXT
        )
    ''')

    conn.commit()
    return conn


def save_event(conn, date, title, category, impact, notes, completed=0):
    c = conn.cursor()
    c.execute('''
        INSERT INTO events (date, title, category, impact, notes, completed)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (date, title, category, impact, notes, completed))
    conn.commit()


def update_event(conn, id, date, title, category, impact, notes, completed):
    c = conn.cursor()
    c.execute('''
        UPDATE events 
        SET date = ?, title = ?, category = ?, impact = ?, notes = ?, completed = ?
        WHERE id = ?
    ''', (date, title, category, impact, notes, completed, id))
    conn.commit()


def delete_event(conn, id):
    c = conn.cursor()
    c.execute('DELETE FROM events WHERE id = ?', (id,))
    conn.commit()


def get_events(conn):
    c = conn.cursor()
    c.execute('SELECT * FROM events ORDER BY date')
    events = c.fetchall()
    return events


def save_score(conn, date, interest_rates, inflation, dollar_strength, supply,
               demand, positioning, notes):
    c = conn.cursor()
    c.execute('''
        INSERT INTO scores (date, interest_rates, inflation, dollar_strength, supply, demand, positioning, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (date, interest_rates, inflation, dollar_strength, supply, demand,
          positioning, notes))
    conn.commit()


def get_scores(conn):
    c = conn.cursor()
    c.execute('SELECT * FROM scores ORDER BY date DESC')
    scores = c.fetchall()
    return scores