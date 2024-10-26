import json
from db_master.db_manager import DatabaseManager

class Player:
    def __init__(self, id, name, profession, inventory, backstory, current_state="neutral"):
        """
        Initializes a player with base attributes including inventory and skill attributes.
        :param id: Unique identifier for the player.
        :param name: Name of the player.
        :param profession: Player's profession or role in the game.
        :param inventory: List of items in the player's possession.
        :param backstory: Brief history or background of the player.
        :param current_state: Player's emotional or physical state.
        """
        self.id = id
        self.name = name
        self.profession = profession
        self.backstory = backstory
        self.current_state = current_state
        self.inventory = inventory
        self.attributes = {"Wisdom": 0, "Courage": 0}  # Skill attributes

    def add_to_db(self):
        """
        Save the current state of the player, including inventory and attributes, to the database.
        """
        db = DatabaseManager()
        query = '''
            INSERT OR REPLACE INTO player (id, name, profession, backstory, current_state, inventory, attributes)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        '''
        data = (
            self.id,
            self.name,
            self.profession,
            self.backstory,
            self.current_state,
            json.dumps(self.inventory),
            json.dumps(self.attributes)
        )
        db.cursor.execute(query, data)
        db.conn.commit()

    def load_from_db(self, player_name):
        """
        Load a player's state from the database based on their name.
        :param player_name: The name of the player to load data for.
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
            self.attributes = json.loads(result['attributes'])

    def improve_attribute(self, attribute, amount=1):
        """
        Increases the value of a specific attribute.
        :param attribute: Name of the attribute to improve.
        :param amount: Amount to increase.
        """
        if attribute in self.attributes:
            self.attributes[attribute] += amount
            print(f"{self.name} has gained {amount} in {attribute} (Total: {self.attributes[attribute]}).")
        else:
            print(f"Attribute {attribute} does not exist.")

    def move(self, prompt):
        """
        Record the player's current action or move in the game.
        :param prompt: Description of the player's action.
        """
        self.current_move = prompt
        print(f"{self.name} decides to: {prompt}")
    
    def add_item_to_inventory(self, item):
        """
        Add an item to the player's inventory.
        :param item: The item to add.
        """
        self.inventory.append(item)
        self.add_to_db()  # Update the database with the new inventory
        print(f"{item} added to {self.name}'s inventory.")

    def remove_item_from_inventory(self, item):
        """
        Remove an item from the player's inventory if it exists.
        :param item: The item to remove.
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
        :return: List of inventory items.
        """
        return self.inventory

    def get_attributes(self):
        """
        Returns a dictionary of the player's attributes.
        :return: Dictionary of attributes (e.g., Wisdom, Courage).
        """
        return self.attributes

    def __str__(self):
        """
        String representation of the player object for easy inspection.
        :return: A string with player details.
        """
        return (f"Player: {self.name}, Profession: {self.profession}, "
                f"State: {self.current_state}, Inventory: {self.inventory}, "
                f"Attributes: {self.attributes}, Last Move: {self.current_move}")
