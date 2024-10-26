import yaml
from llm_master.llm import LLM
from npc_handler import NPC
from quest_handler import Quest
from player_handler import Player
from db_master.db_manager import DatabaseManager
from thechosenone.inference import generate_game_image
from PIL import Image, ImageDraw, ImageFont

class StoryManager:
    def __init__(self, story_file='story.yaml'):
        """
        Initializes the story manager by loading story and NPC details from a YAML file.
        """
        self.db = DatabaseManager()
        self.llm = LLM()
        self.story = self.load_story(story_file)
        self.player = None
        self.npcs = []  # List to hold NPC objects
        self.quests = []  # List to hold quests

    def load_story(self, file_path):
        """Loads the story data from a YAML file."""
        with open(file_path, 'r') as file:
            return yaml.safe_load(file)

    def setup_game(self):
        """Set up player, NPCs, and quests based on story configuration."""
        self.setup_player()
        self.setup_npcs()
        self.setup_quests()

    def setup_player(self):
        """Initialize the player character."""
        player_data = self.story.get('player', {})
        self.player = Player(
            id=player_data.get('id'),
            name=player_data.get('name', 'Unknown Hero'),
            profession=player_data.get('profession', 'Adventurer'),
            inventory=player_data.get('inventory', []),
            backstory=player_data.get('backstory', 'A mysterious figure...'),
            current_state=player_data.get('current_state', 'neutral')
        )
        self.player.add_to_db()

    def setup_quests(self):
        """Set up quests from the story data."""
        quest_data = self.story.get('quests', [])

        for quest_info in quest_data:
            quest = Quest(
                quest_id=quest_info.get('id'),
                name=quest_info.get('name'),
                description=quest_info.get('description'),
                objectives=quest_info.get('objectives', []),
                npcs=quest_info.get('npcs', [])
            )
            self.quests.append(quest)
            quest.add_to_db()
            print(f"Quest '{quest.name}' loaded with objectives: {quest.objectives}")

    def setup_npcs(self):
        """Set up NPCs from the story data."""
        npc_data = self.story.get('npcs', [])

        for npc_info in npc_data:
            npc = NPC(
                id=npc_info.get('id'),
                name=npc_info.get('name'),
                profession=npc_info.get('profession'),
                temperament=npc_info.get('temperament'),
                personality_traits=npc_info.get('personality_traits', []),
                backstory=npc_info.get('backstory'),
                relationship_level=npc_info.get('relationship_level', 0)
            )
            self.npcs.append(npc)
            npc.add_to_db()
            print(f"NPC '{npc.name}' loaded.")

    def start_intro(self):
        """Display the story intro and set the game stage."""
        intro_text = self.story.get('intro', "The adventure begins...")
        return intro_text

    def play_quest(self, quest_id, player_input):
        """
        Handles the player input for a quest and generates appropriate responses from the NPCs,
        including images generated for specific interactions.
        :param quest_id: ID of the current quest.
        :param player_input: Input from the player describing their action or choice.
        :return: List of dictionaries with dialogue and image data for each NPC interaction.
        """
        quest = next((quest for quest in self.quests if quest.quest_id == quest_id), None)
        
        if not quest:
            print("Quest not found.")
            return []

        if quest.status == "completed":
            print(f"Quest '{quest.name}' is already completed.")
            return []

        # Start the quest if it's not already started
        if quest.status == "not_started":
            quest.start()
        
        quest_npcs = [npc for npc in self.npcs if npc.name in quest.npcs]

        # Player attributes improvement based on interaction context
        if "Eldon" in [npc.name for npc in quest_npcs]:
            self.player.improve_attribute("Wisdom", 2)
        if "Lira" in [npc.name for npc in quest_npcs]:
            self.player.improve_attribute("Courage", 1)

        # Initialize response data list for all NPC interactions
        response_data = []

        for npc in quest_npcs:
            # Generate an NPC-specific image based on the current player input or NPC state
            prompt = f"{npc.name} in the setting of {quest.name}, reacting to {player_input}"
            generated_image = generate_game_image(prompt, character_name=npc.name)

            # Generate conditional dialogue based on attributes and interaction history
            npc_response = npc.generate_conditional_response(
                player_attributes=self.player.get_attributes(),
                interaction_history=npc.memory
            )

            # Append each interaction with NPC name, dialogue, and generated image
            response_data.append({"npc": npc.name, "dialogue": npc_response, "image": generated_image})

        # Process player input and dialogue responses
        dialogue_responses = self.llm.generate_response(
            quest_name=quest.name,
            player_action=player_input,
            player_state=self.player.__dict__,
            quest_background=quest.description
        )

        # Check if quest objective is completed
        quest.update_progress(self.player.__dict__, self.llm)

        return response_data

    def run_quest(self, quest):
        """
        Handle quest objectives, including player actions, LLM responses, and visual progression.
        """
        while quest.status == "in_progress":
            current_objective = quest.get_current_objective()
            
            if not current_objective:
                quest.complete(self.llm)
                break

            quest_npcs = [npc for npc in self.npcs if npc.name in quest.npcs]

            # Retrieve memories for prompt engineering
            npc_memory = "\n".join(npc.get_memory_context() for npc in quest_npcs)
            player_memory = self.player.current_move

            # Improve player skills based on interactions with specific NPCs
            if "Eldon" in [npc.name for npc in quest_npcs]:
                self.player.improve_attribute("Wisdom", 2)
            if "Lira" in [npc.name for npc in quest_npcs]:
                self.player.improve_attribute("Courage", 1)

            # Generate dialogue using the enhanced prompt from LLM
            dialogue_response = self.llm.generate_response(
                quest_name=quest.name,
                player_action=self.player.current_move,
                player_state=self.player.__dict__,
                quest_background=quest.description,
                npc_memory=npc_memory,
                player_memory=player_memory
            )

            # Generate conditional NPC responses for the quest interactions
            for npc in quest_npcs:
                npc_response = npc.generate_conditional_response(
                    player_attributes=self.player.get_attributes(),
                    interaction_history=npc.memory
                )
                # Generate scene-specific image for the interaction
                prompt = f"{npc.name} reacting to {self.player.current_move} during {quest.name}"
                image = generate_game_image(prompt, character_name=npc.name)
                
                response_data = {
                    "npc": npc.name,
                    "dialogue": npc_response,
                    "image": image
                }
                yield response_data
