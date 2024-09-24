import sqlite3

class DatabaseManager:
    def __init__(self, db_path="db_master/database/prakriti.db"):
        """Initializes the DatabaseManager with a connection to the SQLite database."""
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

        self.conn.commit()

  

    def get_all_npcs(self):
        """Retrieves all NPCs from the database."""
        self.cursor.execute("SELECT id, name FROM npc")
        return self.cursor.fetchall()
    
    def get_all_quests(self):
        """Retrieves all NPCs from the database."""
        self.cursor.execute("SELECT id, name FROM quest")
        return self.cursor.fetchall()


    def close(self):
        """Closes the database connection."""
        self.conn.close()
