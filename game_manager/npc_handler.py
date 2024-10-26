import json
from db_master.db_manager import DatabaseManager

class NPC:
    def __init__(self, id, name, current_state, profession, temperament, personality_traits, backstory, relationship_level):
        """
        Initialize an NPC with basic attributes like name, profession, temperament, and backstory.
        """
        self.id = id
        self.name = name
        self.profession = profession
        self.temperament = temperament
        self.personality_traits = personality_traits
        self.backstory = backstory
        self.current_state = current_state  # Example state that can be modified based on interactions
        self.relationship_level = relationship_level  # Relationship level with the player
        self.memory = []  # Stores recent interactions with the player
        self.goals = []  # Any specific goals or tasks the NPC may have

    def remember_interaction(self, interaction):
        """
        Store the player's action and the NPC's response in memory.
        Updates the database with the latest memory state.
        """
        self.memory.append(interaction)
        if len(self.memory) > 3:  # Limit to recent 3 interactions for simplicity
            self.memory.pop(0)
        self.add_to_db()

    def add_goal(self, goal):
        """
        Add a goal to the NPC's list of goals.
        """
        self.goals.append(goal)
        self.add_to_db()

    def get_memory_context(self):
        """
        Generate a context string from the NPC's memory to pass to the LLM.
        Provides recent player-NPC interaction history.
        """
        if not self.memory:
            return "No prior interactions."
        
        memory_str = "Previous interactions:\n"
        for interaction in self.memory:
            memory_str += f"Player did: {interaction['player']}, NPC responded: {interaction.get(self.name, '')}\n"
        
        return memory_str

    def add_to_db(self):
        """
        Save the current state of the NPC, including memory and goals, to the database.
        """
        db = DatabaseManager()
        query = '''
            INSERT OR REPLACE INTO npc (id, name, profession, temperament, backstory, current_state, memory, goals, personality_traits, relationship_level)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        '''
        data = (
            self.id,
            self.name,
            self.profession,
            self.temperament,
            self.backstory,
            self.current_state,
            json.dumps(self.memory), 
            json.dumps(self.goals), 
            json.dumps(self.personality_traits),
            self.relationship_level
        )
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
            self.id = result['id']
            self.name = result['name']
            self.profession = result['profession']
            self.temperament = result['temperament']
            self.backstory = result['backstory']
            self.current_state = result['current_state']
            self.memory = json.loads(result['memory'])
            self.goals = json.loads(result['goals'])
            self.personality_traits = json.loads(result['personality_traits'])
            self.relationship_level = result['relationship_level']

    def interact(self, player_input, llm):
        """
        Generate a response from the NPC based on player input, personality, and memory.
        Uses LLM to craft a dynamic and personalized response.
        """
        prompt = f'''
        Act as {self.name}, a {self.profession} with a {self.temperament} temperament. 
        Personality traits: {", ".join(self.personality_traits)}. 
        NPC Memory: {self.get_memory_context()}.
        Player action: {player_input}.
        Respond naturally based on the context and your relationship with the player.
        '''
        response = llm.inference(prompt=prompt)
        interaction = {"player": player_input, self.name: response}
        self.remember_interaction(interaction)
        return response

    def __str__(self):
        """
        String representation of the NPC object for easy inspection.
        :return: A string with NPC details
        """
        return (f"NPC: {self.name}, Profession: {self.profession}, Temperament: {self.temperament}, "
                f"Current State: {self.current_state}, Relationship Level: {self.relationship_level}, "
                f"Personality Traits: {', '.join(self.personality_traits)}, Memory: {self.get_memory_context()}")
