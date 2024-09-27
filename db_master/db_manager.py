import sqlite3

class DatabaseManager:
    def __init__(self, db_path="db_master/database/prakriti.db"):
        """
        Initializes the DatabaseManager with a connection to the SQLite database.
        """
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.create_tables()

    def create_tables(self):
        """Creates the necessary tables in the database."""

        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS npc (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                profession TEXT,
                temperament TEXT,
                backstory TEXT,
                current_state TEXT DEFAULT 'neutral',
                memory TEXT,  -- JSON or serialized text to store memory of interactions
                emotion_history TEXT,  -- JSON or serialized text to store emotional history
                goals TEXT,  -- JSON or serialized text to store goals
                relationship_level INTEGER DEFAULT 0
            )
        ''')

        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS quest (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                description TEXT,
                objectives TEXT,
                status TEXT
            )
        ''')

        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS story_progress (
                key TEXT PRIMARY KEY,
                value TEXT  -- Store key-value pairs for player story progress and decisions
            )
        ''')

        self.conn.commit()

    def insert_npc(self, npc_data):
        """Inserts or updates an NPC in the database."""
        query = '''
            INSERT OR REPLACE INTO npc (id, name, profession, temperament, backstory, current_state, memory, emotion_history, goals, relationship_level)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        '''
        self.cursor.execute(query, npc_data)
        self.conn.commit()

    def fetch_npc(self, npc_id):
        """Fetches an NPC from the database based on the NPC's ID."""
        self.cursor.execute("SELECT * FROM npc WHERE id=?", (npc_id,))
        return self.cursor.fetchone()
    
    def insert_quest(self, quest_data):
        """Inserts or updates a quest in the database."""
        query = '''
            INSERT OR REPLACE INTO quest (id, name, description, objectives, status)
            VALUES (?, ?, ?, ?, ?)
        '''
        self.cursor.execute(query, quest_data)
        self.conn.commit()
    
    def fetch_quest(self, quest_id):
        """Fetches a quest from the database based on the quest's ID."""
        self.cursor.execute("SELECT * FROM quest WHERE id=?", (quest_id,))
        return self.cursor.fetchone()
    
    def update_story_progress(self, key, value):
        """Updates or inserts the player's story progress in the database."""
        query = '''
            INSERT OR REPLACE INTO story_progress (key, value)
            VALUES (?, ?)
        '''
        self.cursor.execute(query, (key, value))
        self.conn.commit()

    def fetch_story_progress(self, key):
        """Fetches a specific story progress key from the database."""
        self.cursor.execute("SELECT value FROM story_progress WHERE key=?", (key,))
        return self.cursor.fetchone()


    def close(self):
        """Closes the database connection."""
        self.conn.close()
