import json
from db_master.db_manager import DatabaseManager

class Player:
    def __init__(self, id,  name, current_state, profession, inventory, backstory):
        """
        Initialize an NPC with basic attributes like name, profession, temperament, and backstory.
        """
        self.id = id
        self.name = name
        self.profession = profession
        self.backstory = backstory
        self.current_state = current_state  # Example state that can be modified
        self.inventory = inventory
        self.current_move = None
    
    def add_to_db(self):
        db = DatabaseManager()

        query = '''
            INSERT OR REPLACE INTO player (name, profession, backstory, current_state, inventory)
            VALUES (?, ?, ?, ?,?);
        '''

        data = (
            self.name, 
            self.profession, 
            self.backstory, 
            self.current_state,
            self.inventory)
        
        db.cursor.execute(query, data)
        db.conn.commit()


    def load_from_db(self, player_name):
        """
        Load an NPC's state from the database based on their name.
        """
        db = DatabaseManager()
        query = "SELECT * FROM player WHERE name = ?"
        result = db.cursor.execute(query, (player_name,)).fetchone()
        
        if result:
            self.name = result['name']
            self.profession = result['profession']
            self.backstory = result['backstory']
            self.current_state = result['current_state']
    
    def move(self, prompt):
        self.current_move = "player:" + prompt

    def __str__(self):
        """
        String representation of the player object for easy inspection.
        :return: A string with player details
        """
        return f"Player: {self.name}, Profession: {self.profession}, State: {self.current_state}, Inventory: {self.inventory}, current prompt move of the player: {self.current_move}"
