# story_engine.py
from db_manager import DBManager
from llm_engine import LLMEngine

class StoryEngine:
    def __init__(self):
        # Initialize LLM and DB manager
        self.llm = LLMEngine()
        self.db = DBManager(password="npc@prakriti")  # Replace with your Neo4j password
        
        # Start the game at the initial scene
        self.current_scene_id = 1
        self.current_context = self.db.get_scene_by_id(self.current_scene_id)

    def update_scene(self, new_scene_id):
        """
        Update to a new scene based on the player's choice.
        """
        self.current_scene_id = new_scene_id
        self.current_context = self.db.get_scene_by_id(new_scene_id)

    def get_next_scene_options(self):
        """
        Get possible next scenes for the current scene.
        """
        return self.db.get_next_scene_options(self.current_scene_id)

    def process_player_input(self, player_input):
        """
        Generate NPC response based on player input and the current scene context.
        """
        return self.llm.generate_response(player_input, self.current_context)
