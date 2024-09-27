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
        self.emotion_history = []
        self.personality_traits = personality_traits
        self.relationship_level = relationship_level # Track relationship level with the player (e.g., positive, negative)


    def remember_interaction(self, player_action, npc_response):
        """
        Store the player's action and the NPC's response in memory.
        """
        interaction = {
            "player_action": player_action,
            "npc_response": npc_response
        }
        self.memory.append(interaction)
        
        # Update the database with the latest memory state
        self.add_to_db()


    def update_emotion(self, new_state):
        """
        Update the NPC's emotional state and track its history.
        """
        self.emotion_history.append(self.current_state)
        self.current_state = new_state
        self.add_to_db()

    def add_goal(self, goal):
        """
        Add a goal to the NPC's list of goals.
        :param goal: The goal the NPC wants to achieve
        """
        self.goals.append(goal)
        self.add_to_db()

    def modify_relationship(self, change):
        """
        Modify the relationship level between the NPC and the player.
        :param change: The change to apply to the relationship level (positive or negative)
        """
        self.relationship_level += change
        self.add_to_db()

    def get_memory_context(self):
        # Generate a context string from the NPC's memory to pass to the LLM
        if not self.memory:
            return "No prior interactions."
        
        memory_str = "Previous interactions:\n"
        for interaction in self.memory[-3:]:  # Limit to the last 3 interactions
            memory_str += f"Player did: {interaction['player_action']}, NPC responded: {interaction['npc_response']}\n"
        
        return memory_str
    
    def add_to_db(self):
        db = DatabaseManager()

        query = '''
            INSERT OR REPLACE INTO npc (name, profession, temperament, backstory, current_state, memory, emotion_history, goals, relationship_level)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
        '''

        data = (self.name, self.profession, self.temperament, self.backstory, self.current_state, json.dumps(self.memory), json.dumps(self.emotion_history), json.dumps(self.goals), self.relationship_level)
        
        db.cursor.execute(query, data)
    


    def __str__(self):
        """
        String representation of the NPC object for easy inspection.
        :return: A string with NPC details
        """
        return f"NPC: {self.name}, Profession: {self.profession}, Temperament: {self.temperament}, State: {self.current_state}, Relationship Level: {self.relationship_level}"
