import yaml
from llm_master.llm import LLM
from npc_handler import NPC
from quest_handler import Quest
from db_master.db_manager import DatabaseManager


class StoryManager:
    def __init__(self, story_file='story.yaml'):
        """
        Initializes the story manager by loading story and NPC details from a YAML file.
        """
        self.db = DatabaseManager()
        self.story = self.load_story(story_file)
        self.player = None  # Placeholder for player character
        self.npcs = []  # List to hold NPC objects
        self.quests = []  # List to hold quests
        self.llm = LLM(args=None) 

    def load_story(self, file_path):
        """Loads the story data from a YAML file."""
        with open(file_path, 'r') as file:
            return yaml.safe_load(file)

    def setup_game(self):
        """Set up player, NPCs, and quests."""
        self.setup_player()
        self.setup_npcs()
        self.setup_quests()

    def setup_player(self):
        """Initialize the player character."""
        player_data = self.story.get('player', {})
        player_name = player_data.get('name', 'Unknown Hero')
        player_backstory = player_data.get('backstory', 'A mysterious figure...')
        
        self.player = {
            'name': player_name,
            'backstory': player_backstory
        }
        

    def setup_npcs(self):
        """Creates NPCs based on story data."""
        npc_data = self.story.get('npcs', [])
        
        for npc_info in npc_data:
            npc = NPC(
                id=npc_info.get('id'),
                name=npc_info.get('name'),
                current_state=npc_info.get('current_state', 'neutral'),
                profession=npc_info.get('profession', 'None'),
                temperament=npc_info.get('temperament', 'calm'),
                personality_traits=npc_info.get('personality_traits', []),
                backstory=npc_info.get('backstory', 'Unknown history'),
                relationship_level=npc_info.get('relationship_level', 0)
            )
            self.npcs.append(npc)
            npc.add_to_db()  # Store NPC in database


    def setup_quests(self):
        """Set up quests from the story data."""
        quest_data = self.story.get('quests', [])
        for quest_info in quest_data:
            quest = Quest(
                quest_id=quest_info.get('id'),
                name=quest_info.get('name'),
                description=quest_info.get('description'),
                objectives=quest_info.get('objectives', [])
            )
            self.quests.append(quest)
            quest.add_to_db()  # Store quest in database


    def intro(self):
        """Print the introductory story and set the stage for gameplay."""
        intro_text = self.story.get('intro', "The adventure begins...")
        print(intro_text)
        self.llm_response()  # Call the LLM for a dynamic introduction
   
    def play(self):
        """Main gameplay loop."""
        self.intro()
        while True:
            command = input("What do you want to do? (interact, quest, quit): ").strip().lower()
            if command == 'interact':
                npc_id = int(input("Enter NPC ID to interact with: "))
                self.interact_with_npc(npc_id)
            elif command == 'quest':
                quest_id = int(input("Enter Quest ID to start: "))
                self.start_quest(quest_id)
            elif command == 'quit':
                print("Thanks for playing!")
                break
            else:
                print("Invalid command. Please try again.")

    def play(self):
        self.intro()