# db_utils.py
import sqlite3
from datetime import datetime
import os
import streamlit as st



# Cache the database connection
@st.cache_resource
def get_connection():
    """Create a database connection that can be reused across reruns."""
    try:
        db_path = "/tmp/gold_analysis.db"
        conn = sqlite3.connect(db_path)
        return conn
    except Exception as e:
        st.error(f"Failed to connect to database: {str(e)}")
        return None

# Then replace your init_db function with this
def init_db():
    """Initialize the database schema with proper tables."""
    conn = get_connection()
    if conn is None:
        return None
        
    try:
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
    except Exception as e:
        st.error(f"Database initialization error: {str(e)}")
        return None


def save_event(conn, date, title, category, impact, notes, completed=0):
    """Save an event with proper error handling."""
    if conn is None:
        st.error("Database connection not available")
        return False
    
    try:
        c = conn.cursor()
        c.execute('''
            INSERT INTO events (date, title, category, impact, notes, completed)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (date, title, category, impact, notes, completed))
        conn.commit()
        return True
    except Exception as e:
        st.error(f"Error saving event: {str(e)}")
        return False


def update_event(conn, id, date, title, category, impact, notes, completed):
    """Update an event with proper error handling."""
    if conn is None:
        st.error("Database connection not available")
        return False
    
    try:
        c = conn.cursor()
        c.execute('''
            UPDATE events 
            SET date = ?, title = ?, category = ?, impact = ?, notes = ?, completed = ?
            WHERE id = ?
        ''', (date, title, category, impact, notes, completed, id))
        conn.commit()
        return True
    except Exception as e:
        st.error(f"Error updating event: {str(e)}")
        return False


def delete_event(conn, id):
    """Delete an event with proper error handling."""
    if conn is None:
        st.error("Database connection not available")
        return False
    
    try:
        c = conn.cursor()
        c.execute('DELETE FROM events WHERE id = ?', (id,))
        conn.commit()
        return True
    except Exception as e:
        st.error(f"Error deleting event: {str(e)}")
        return False


def get_events(conn):
    """Get all events with proper error handling."""
    if conn is None:
        st.error("Database connection not available")
        return []
    
    try:
        c = conn.cursor()
        c.execute('SELECT * FROM events ORDER BY date')
        events = c.fetchall()
        return events
    except Exception as e:
        st.error(f"Error getting events: {str(e)}")
        return []


def save_score(conn, date, interest_rates, inflation, dollar_strength, supply,
               demand, positioning, notes):
    """Save a score with proper error handling."""
    if conn is None:
        st.error("Database connection not available")
        return False
    
    try:
        c = conn.cursor()
        c.execute('''
            INSERT INTO scores (date, interest_rates, inflation, dollar_strength, supply, demand, positioning, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (date, interest_rates, inflation, dollar_strength, supply, demand,
              positioning, notes))
        conn.commit()
        return True
    except Exception as e:
        st.error(f"Error saving score: {str(e)}")
        return False


def get_scores(conn):
    """Get all scores with proper error handling."""
    if conn is None:
        st.error("Database connection not available")
        return []
    
    try:
        c = conn.cursor()
        c.execute('SELECT * FROM scores ORDER BY date DESC')
        scores = c.fetchall()
        return scores
    except Exception as e:
        st.error(f"Error getting scores: {str(e)}")
        return []