# dialogue_manager.py
from story_engine import StoryEngine

class DialogueManager:
    def __init__(self):
        # Initialize the story engine
        self.story_engine = StoryEngine()

    def handle_dialogue(self, player_input):
        """
        Handle player dialogue, retrieve NPC response, and manage context updates.

        Args:
            player_input (str): The player's dialogue input.

        Returns:
            str: NPC's response based on the current context and player input.
        """
        npc_response = self.story_engine.process_player_input(player_input)
        return npc_response

    def get_next_scenes(self):
        """
        Retrieve next scene options.
        """
        return self.story_engine.get_next_scene_options()

    def change_scene(self, scene_id):
        """
        Update the current scene based on player's choice.
        """
        self.story_engine.update_scene(scene_id)
