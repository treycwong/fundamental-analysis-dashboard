# db_utils.py
import sqlite3
from datetime import datetime
import os
import streamlit as st
import json

# Cache the database connection
@st.cache_resource(ttl="1h")
def get_connection():
    """Create a database connection that can be reused across reruns."""
    try:
        db_path = "/tmp/gold_analysis.db"
        conn = sqlite3.connect(db_path, check_same_thread=False)  # Important: allow usage from different threads
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

        c.execute('''
            CREATE TABLE IF NOT EXISTS claude_analyses (
                id INTEGER PRIMARY KEY,
                date TEXT,
                analysis_type TEXT,
                data TEXT,
                created_at TEXT
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
    
def save_claude_analysis(conn, analysis_type, data):
    """Save Claude's analysis to the database"""
    if conn is None:
        return False
    
    try:
        today = datetime.now().strftime("%Y-%m-%d")
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        c = conn.cursor()
        
        # First check if an analysis exists for today
        c.execute('SELECT id FROM claude_analyses WHERE date = ? AND analysis_type = ?', 
                 (today, analysis_type))
        existing = c.fetchone()
        
        if existing:
            # Update existing analysis
            c.execute('''
                UPDATE claude_analyses 
                SET data = ?, created_at = ?
                WHERE id = ?
            ''', (json.dumps(data), now, existing[0]))
        else:
            # Insert new analysis
            c.execute('''
                INSERT INTO claude_analyses (date, analysis_type, data, created_at)
                VALUES (?, ?, ?, ?)
            ''', (today, analysis_type, json.dumps(data), now))
        
        conn.commit()
        return True
    except Exception as e:
        st.error(f"Error saving Claude analysis: {str(e)}")
        return False

def get_latest_claude_analysis(conn, analysis_type):
    """Get the latest Claude analysis from the database"""
    if conn is None:
        return None, None
    
    try:
        c = conn.cursor()
        c.execute('''
            SELECT data, created_at FROM claude_analyses 
            WHERE analysis_type = ?
            ORDER BY date DESC, created_at DESC LIMIT 1
        ''', (analysis_type,))
        result = c.fetchone()
        
        if result:
            data = json.loads(result[0])
            created_at = result[1]
            return data, created_at
        return None, None
    except Exception as e:
        st.error(f"Error retrieving Claude analysis: {str(e)}")
        return None, None