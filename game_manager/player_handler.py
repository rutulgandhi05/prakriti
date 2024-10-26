import json
from db_master.db_manager import DatabaseManager

class Player:
    def __init__(self, id, name, current_state, profession, inventory, backstory):
        """
        Initialize the player with basic attributes like name, profession, inventory, and backstory.
        """
        self.id = id
        self.name = name
        self.profession = profession
        self.backstory = backstory
        self.current_state = current_state  # Example state that can be modified based on interactions
        self.inventory = inventory
        self.current_move = None  # Last recorded action of the player

    def add_to_db(self):
        """
        Save the current state of the player, including inventory and current state, to the database.
        """
        db = DatabaseManager()
        query = '''
            INSERT OR REPLACE INTO player (id, name, profession, backstory, current_state, inventory)
            VALUES (?, ?, ?, ?, ?, ?);
        '''
        data = (
            self.id,
            self.name,
            self.profession,
            self.backstory,
            self.current_state,
            json.dumps(self.inventory)
        )
        db.cursor.execute(query, data)
        db.conn.commit()

    def load_from_db(self, player_name):
        """
        Load a player's state from the database based on their name.
        """
        db = DatabaseManager()
        query = "SELECT * FROM player WHERE name = ?"
        result = db.cursor.execute(query, (player_name,)).fetchone()
        
        if result:
            self.id = result['id']
            self.name = result['name']
            self.profession = result['profession']
            self.backstory = result['backstory']
            self.current_state = result['current_state']
            self.inventory = json.loads(result['inventory'])

    def move(self, prompt):
        """
        Record the player's current action.
        """
        self.current_move = prompt
        print(f"{self.name} decides to: {prompt}")
    
    def add_item_to_inventory(self, item):
        """
        Add an item to the player's inventory.
        """
        self.inventory.append(item)
        self.add_to_db()  # Update the database with the new inventory
        print(f"{item} added to {self.name}'s inventory.")

    def remove_item_from_inventory(self, item):
        """
        Remove an item from the player's inventory if it exists.
        """
        if item in self.inventory:
            self.inventory.remove(item)
            self.add_to_db()  # Update the database with the modified inventory
            print(f"{item} removed from {self.name}'s inventory.")
        else:
            print(f"{item} is not in {self.name}'s inventory.")

    def get_inventory(self):
        """
        Returns the player's current inventory.
        """
        return self.inventory

    def __str__(self):
        """
        String representation of the player object for easy inspection.
        :return: A string with player details.
        """
        return (f"Player: {self.name}, Profession: {self.profession}, State: {self.current_state}, "
                f"Inventory: {self.inventory}, Last Move: {self.current_move}")
