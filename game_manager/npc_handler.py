import json
from db_master.db_manager import DatabaseManager

class NPC:
    def __init__(self, id,  name, current_state, profession, temperament, personality_traits, backstory, relationship_level):
        """
        Initialize an NPC with basic attributes like name, profession, temperament, and backstory.
        """
        self.id = id
        self.name = name
        self.profession = profession
        self.temperament = temperament
        self.backstory = backstory
        self.memory = []  # Stores past interactions with the player
        self.current_state = current_state  # Example state that can be modified
        self.goals = []
        self.personality_traits = personality_traits


    def remember_interaction(self, interaction):
        """
        Store the player's action and the NPC's response in memory.
        """
        
        self.memory.append(interaction)
        
        # Update the database with the latest memory state
        self.add_to_db()


    def add_goal(self, goal):
        """
        Add a goal to the NPC's list of goals.
        :param goal: The goal the NPC wants to achieve
        """
        self.goals.append(goal)
        self.add_to_db()


    def get_memory_context(self):
        # Generate a context string from the NPC's memory to pass to the LLM
        if not self.memory:
            return "No prior interactions."
        
        memory_str = "Previous interactions:\n"
        for interaction in self.memory[-4:]:  # Limit to the last 3 interactions
            memory_str += f"Player did: {interaction['player']}, NPC responded: {interaction[{self.name}]}\n"
        
        return memory_str
    

    def add_to_db(self):
        db = DatabaseManager()

        query = '''
            INSERT OR REPLACE INTO npc (name, profession, temperament, backstory, current_state, memory, goals, personality_traits)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?);
        '''

        data = (
            self.name, 
            self.profession, 
            self.temperament, 
            self.backstory, 
            self.current_state,
            json.dumps(self.memory), 
            json.dumps(self.goals), 
            self.personality_traits)
        
        db.cursor.execute(query, data)
        db.conn.commit()


    def load_from_db(self, npc_name):
        """
        Load an NPC's state from the database based on their name.
        """
        db = DatabaseManager()
        query = "SELECT * FROM npc WHERE name = ?"
        result = db.cursor.execute(query, (npc_name,)).fetchone()
        
        if result:
            self.name = result['name']
            self.profession = result['profession']
            self.temperament = result['temperament']
            self.backstory = result['backstory']
            self.current_state = result['current_state']
            self.memory = json.loads(result['memory'])
            self.goals = json.loads(result['goals'])
            self.personality_traits = result['personality_traits']
            

    def __str__(self):
        """
        String representation of the NPC object for easy inspection.
        :return: A string with NPC details
        """
        return f"NPC: {self.name}, Profession: {self.profession}, Temperament: {self.temperament}, State: {self.current_state}, personality_traits : {self.personality_traits}, memory: {self.get_memory_context()}"
