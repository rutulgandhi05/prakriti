import sqlite3
import json

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
                summarized_memory TEXT,    -- Summary of old interactions (compressed or summarized)
                recent_memory TEXT         -- JSON of recent interactions (e.g., last 2-3)
                goals TEXT,  -- JSON or serialized text to store goals
                personality_traits TEXT,
            )
        ''')

        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS quest (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                description TEXT,
                objectives TEXT,
                status TEXT,
                objective_index TEXT,
            )
        ''')

        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS story_progress (
                key TEXT PRIMARY KEY,
                value TEXT  -- Store key-value pairs for player story progress and decisions
            )
        ''')

        self.conn.commit()

    def insert_npc(self, npc):
        """Inserts or updates an NPC in the database."""
        query = '''
            INSERT OR REPLACE INTO npc (id, name, profession, temperament, backstory, current_state, memory, emotion_history, goals, relationship_level)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        '''
        self.cursor.execute(query, npc)
        self.conn.commit()


    def fetch_npc(self, npc_id):
        """
        Fetches an NPC from the database based on the NPC's ID.
        :param npc_id: The ID of the NPC to fetch.
        :return: A dictionary containing the NPC data or None if not found.
        """
        self.cursor.execute("SELECT * FROM npc WHERE id=?", (npc_id,))
        row = self.cursor.fetchone()
        
        if row:
            return {
                "id": row[0],
                "name": row[1],
                "profession": row[2],
                "temperament": row[3],
                "backstory": row[4],
                "current_state": row[5],
                "memory": json.loads(row[6]),
                "emotion_history": json.loads(row[7]),
                "goals": json.loads(row[8]),
                "relationship_level": row[9]
            }
        return None
    
    def insert_quest(self, quest_data):
        """Inserts or updates a quest in the database."""
        query = '''
            INSERT OR REPLACE INTO quest (id, name, description, objectives, status)
            VALUES (?, ?, ?, ?, ?)
        '''
        self.cursor.execute(query, quest_data)
        self.conn.commit()
    
    
    def fetch_quest(self, quest_id):
        """
        Fetches a quest from the database based on the quest's ID.
        :param quest_id: The ID of the quest to fetch.
        :return: A dictionary containing the quest data or None if not found.
        """
        self.cursor.execute("SELECT * FROM quest WHERE id=?", (quest_id,))
        row = self.cursor.fetchone()
        
        if row:
            return {
                "id": row[0],
                "name": row[1],
                "description": row[2],
                "objectives": json.loads(row[3]),
                "status": row[4]
            }
        return None
    
    
    def update_story_progress(self, key, value):
            """
            Updates or inserts the player's story progress in the database.
            :param key: The key representing the progress (e.g., 'quest_1_complete').
            :param value: The value associated with the progress key.
            """
            query = '''
                INSERT OR REPLACE INTO story_progress (key, value)
                VALUES (?, ?)
            '''
            self.cursor.execute(query, (key, value))
            self.connection.commit()


    def fetch_story_progress(self, key):
        """
        Fetches a specific story progress key from the database.
        :param key: The key representing the story progress.
        :return: The value associated with the key, or None if not found.
        """
        self.cursor.execute("SELECT value FROM story_progress WHERE key=?", (key,))
        result = self.cursor.fetchone()
        return result[0] if result else None


    def save_npc_state(self, npc):
        """
        Saves the current state of the NPC to the database.
        :param npc: An NPC object with attributes like memory, emotion history, and goals.
        """
        npc_data = (
            npc.id,
            npc.name,
            npc.profession,
            npc.temperament,
            npc.backstory,
            npc.current_state,
            json.dumps(npc.memory),
            json.dumps(npc.emotion_history),
            json.dumps(npc.goals),
            npc.relationship_level
        )
        self.insert_npc(npc_data)


    def save_quest_progress(self, quest_id, status):
        """
        Saves the progress of a quest to the database.
        :param quest_id: The ID of the quest.
        :param status: The current status of the quest (e.g., 'in_progress', 'completed').
        """
        query = "UPDATE quest SET status = ? WHERE id = ?"
        self.cursor.execute(query, (status, quest_id))
        self.connection.commit()


    def close(self):
        """Closes the database connection."""
        self.conn.close()
