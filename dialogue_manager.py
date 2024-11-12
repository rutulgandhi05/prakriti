import logging
from db_manager import DBManager
from llm_engine import LLMEngine

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class DialogueManager:
    def __init__(self):
        logging.info("Initializing DialogueManager")
        self.db_manager = DBManager()   
        self.llm_engine = LLMEngine()   


    def close(self):
        """
        Close resources used by the DialogueManager.
        """
        self.db_manager.close()

    def handle_dialogue(self, player_input, scene_id):
        logging.info(f"Handling dialogue for scene ID: {scene_id} with player input: {player_input}")
        # Retrieve scene description and NPCs for the current scene
        description, npcs = self.db_manager.get_scene_by_id(scene_id)
        
        npc_name = npcs[0] if npcs else "a character"  # Use first NPC's name or a generic term
        prompt = self.llm_engine.generate_npc_prompt(description, npc_name, player_input)
        
        # Generate response from LLM
        npc_response = self.llm_engine.generate_response(prompt)
        
        return npc_response.replace(prompt, '')

    def generate_enhanced_description(self, basic_description):
        # Prompt the LLM to expand the basic description

        prompt = self.llm_engine.generate_scene_description_prompt(basic_description)
        
        # Generate the enhanced description
        enhanced_description = self.llm_engine.generate_response(prompt)
        
        return enhanced_description.replace(prompt, '')

    def check_scene_transition_conditions(self, player_input, current_scene_description):
        prompt = self.llm_engine.generate_transition_prompt(current_scene_description, player_input)
        transition_response = self.llm_engine.generate_response(prompt)
        return transition_response