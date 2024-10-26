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
        """Creates the necessary tables in the database if they do not exist."""
        # Table for storing NPC information
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS npc (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                profession TEXT,
                temperament TEXT,
                backstory TEXT,
                current_state TEXT DEFAULT 'neutral',
                memory TEXT,               -- JSON of recent interactions
                goals TEXT,                -- JSON to store goals
                personality_traits TEXT,   -- JSON to store personality traits
                relationship_level INTEGER -- Relationship level with player
            )
        ''')

        # Table for storing quest information
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS quest (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                description TEXT,
                objectives TEXT,           -- JSON of quest objectives
                status TEXT,
                objective_index INTEGER DEFAULT 0
            )
        ''')

        # Table for storing player information
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS player (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                profession TEXT,
                backstory TEXT,
                current_state TEXT,
                inventory TEXT             -- JSON of player's inventory items
            )
        ''')

        # Table for tracking story progress or player decisions
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS story_progress (
                key TEXT PRIMARY KEY,
                value TEXT                 -- Stores key-value pairs for story progress
            )
        ''')

        self.conn.commit()

    def insert_npc(self, npc_data):
        """Inserts or updates an NPC in the database."""
        query = '''
            INSERT OR REPLACE INTO npc (id, name, profession, temperament, backstory, current_state, memory, goals, personality_traits, relationship_level)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        '''
        self.cursor.execute(query, npc_data)
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
                "memory": json.loads(row[6]) if row[6] else [],
                "goals": json.loads(row[7]) if row[7] else [],
                "personality_traits": json.loads(row[8]) if row[8] else [],
                "relationship_level": row[9]
            }
        return None

    def insert_quest(self, quest_data):
        """Inserts or updates a quest in the database."""
        query = '''
            INSERT OR REPLACE INTO quest (id, name, description, objectives, status, objective_index)
            VALUES (?, ?, ?, ?, ?, ?)
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
                "objectives": json.loads(row[3]) if row[3] else [],
                "status": row[4],
                "objective_index": row[5]
            }
        return None

    def insert_player(self, player_data):
        """Inserts or updates a player in the database."""
        query = '''
            INSERT OR REPLACE INTO player (id, name, profession, backstory, current_state, inventory)
            VALUES (?, ?, ?, ?, ?, ?)
        '''
        self.cursor.execute(query, player_data)
        self.conn.commit()

    def fetch_player(self, player_id):
        """
        Fetches a player from the database based on the player's ID.
        :param player_id: The ID of the player to fetch.
        :return: A dictionary containing the player data or None if not found.
        """
        self.cursor.execute("SELECT * FROM player WHERE id=?", (player_id,))
        row = self.cursor.fetchone()
        
        if row:
            return {
                "id": row[0],
                "name": row[1],
                "profession": row[2],
                "backstory": row[3],
                "current_state": row[4],
                "inventory": json.loads(row[5]) if row[5] else []
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
        self.conn.commit()

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
        :param npc: An NPC object with attributes like memory, goals, and personality traits.
        """
        npc_data = (
            npc.id,
            npc.name,
            npc.profession,
            npc.temperament,
            npc.backstory,
            npc.current_state,
            json.dumps(npc.memory),
            json.dumps(npc.goals),
            json.dumps(npc.personality_traits),
            npc.relationship_level
        )
        self.insert_npc(npc_data)

    def save_quest_progress(self, quest_id, status, objective_index):
        """
        Saves the progress of a quest to the database.
        :param quest_id: The ID of the quest.
        :param status: The current status of the quest (e.g., 'in_progress', 'completed').
        :param objective_index: The index of the current objective.
        """
        query = "UPDATE quest SET status = ?, objective_index = ? WHERE id = ?"
        self.cursor.execute(query, (status, objective_index, quest_id))
        self.conn.commit()

    def close(self):
        """Closes the database connection."""
        self.conn.close()
