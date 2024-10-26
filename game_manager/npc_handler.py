import json
from db_master.db_manager import DatabaseManager

class NPC:
    def __init__(self, id, name, profession, temperament, personality_traits, backstory, relationship_level=0):
        """
        Initializes an NPC with attributes including relationship level, backstory, and temperament.
        :param id: Unique identifier for the NPC.
        :param name: Name of the NPC.
        :param profession: Profession or role of the NPC in the game.
        :param temperament: General disposition of the NPC (e.g., friendly, cautious).
        :param personality_traits: List of personality traits describing the NPC.
        :param backstory: Brief history or background of the NPC.
        :param relationship_level: Initial relationship level with the player.
        """
        self.id = id
        self.name = name
        self.profession = profession
        self.temperament = temperament
        self.personality_traits = personality_traits
        self.backstory = backstory
        self.current_state = "neutral"
        self.relationship_level = relationship_level  # Relationship level with the player
        self.memory = []  # Stores recent interactions with the player

    def remember_interaction(self, interaction):
        """
        Store the player's action and the NPC's response in memory, adjusting the NPC's relationship level.
        :param interaction: Dictionary describing the interaction with the player.
        """
        self.memory.append(interaction)
        if len(self.memory) > 3:  # Limit to recent 3 interactions for simplicity
            self.memory.pop(0)
        self.adjust_relationship(interaction)
        self.add_to_db()

    def adjust_relationship(self, interaction):
        """
        Adjust the NPC's relationship level based on the type of interaction.
        :param interaction: Dictionary containing details of the interaction.
        """
        if "helped" in interaction.get("player_action", "").lower():
            self.relationship_level += 1
        elif "argued" in interaction.get("player_action", "").lower():
            self.relationship_level -= 1

    def relationship_level_description(self):
        """Provide descriptive text for the NPC's relationship level with the player."""
        if self.relationship_level > 5:
            return "respect and admiration"
        elif self.relationship_level > 0:
            return "a friendly demeanor"
        elif self.relationship_level < 0:
            return "a cautious tone"
        else:
            return "neutrality"

    def generate_conditional_response(self, player_attributes, interaction_history):
        """
        Generate a personalized response based on the player's attributes and past interactions.
        :param player_attributes: Dictionary of player’s skill attributes (e.g., Wisdom, Courage).
        :param interaction_history: List of key past interactions with the player.
        :return: Conditional response string.
        """
        base_response = f"{self.name} looks at you with {self.relationship_level_description()}."

        # Tailor NPC response based on player attributes
        if "Wisdom" in player_attributes and player_attributes["Wisdom"] > 5:
            base_response += " They seem impressed by your insight."
        if "Courage" in player_attributes and player_attributes["Courage"] > 3:
            base_response += " They respect your bravery from past encounters."

        # Reference notable past interactions if any
        if interaction_history:
            base_response += " They recall your actions during your last encounter and adjust their tone accordingly."

        return base_response

    def add_to_db(self):
        """
        Save the current state of the NPC, including memory and relationship level, to the database.
        """
        db = DatabaseManager()
        query = '''
            INSERT OR REPLACE INTO npc (id, name, profession, temperament, backstory, current_state, memory, personality_traits, relationship_level)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        '''
        data = (
            self.id,
            self.name,
            self.profession,
            self.temperament,
            self.backstory,
            self.current_state,
            json.dumps(self.memory), 
            json.dumps(self.personality_traits),
            self.relationship_level
        )
        db.cursor.execute(query, data)
        db.conn.commit()

    def get_memory_context(self):
        """
        Generate a context string from the NPC's memory to pass to the LLM.
        Provides recent player-NPC interaction history.
        :return: String of recent interactions.
        """
        if not self.memory:
            return "No prior interactions."
        
        memory_str = "Previous interactions:\n"
        for interaction in self.memory:
            memory_str += f"Player did: {interaction['player_action']}, NPC responded: {interaction.get(self.name, '')}\n"
        
        return memory_str

    def __str__(self):
        """
        String representation of the NPC object for easy inspection.
        :return: A string with NPC details.
        """
        return (f"NPC: {self.name}, Profession: {self.profession}, Temperament: {self.temperament}, "
                f"Relationship Level: {self.relationship_level}, Personality Traits: {', '.join(self.personality_traits)}, "
                f"Memory: {self.get_memory_context()}")
