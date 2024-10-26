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
                current_state=npc_info.get('current_state', 'neutral'),
                relationship_level=npc_info.get('relationship_level', 0)
            )
            self.npcs.append(npc)
            npc.add_to_db()
            print(f"NPC '{npc.name}' loaded.")

    def start_intro(self):
        """Display the story intro and set the game stage."""
        intro_text = self.story.get('intro', "The adventure begins...")
        return intro_text

    def play_quest(self, quest_id):
        """Plays the quest by handling logic, progression, and visuals."""
        quest = next((quest for quest in self.quests if quest.quest_id == quest_id), None)
        
        if quest:
            if quest.status == "completed":
                print(f"Quest '{quest.name}' is already completed.")
                return

            # Start the quest if it's not started yet
            if quest.status == "not_started":
                quest.start()
            
            # Run through the quest objectives
            self.run_quest(quest)

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

            # Generate dialogue using the enhanced prompt from LLM
            dialogue_response = self.llm.generate_response(
                quest_name=quest.name,
                player_action=self.player.current_move,
                player_state=self.player.__dict__,
                quest_background=quest.description,
                npc_memory=npc_memory,
                player_memory=player_memory
            )

            # Process the dialogue response
            speaker = self.get_speaker(dialogue_response, quest_npcs)
            dialogue = dialogue_response
            interaction = {"player": self.player.current_move, speaker.name: dialogue}

            if speaker.name != "Narrator":
                speaker.remember_interaction(interaction)

            # Generate visual prompt for current scene
            visual_prompt = f'''
            Act as a visual prompt designer. 
            Create a scene prompt for the SDXL model based on current quest state:
            - Quest details: {quest.__str__()}.
            - NPCs present: {[npc.__str__() for npc in quest_npcs]}.
            - Current objective: {current_objective['description']}.
            Player action: {self.player.current_move}.
            Only include NPCs visible to the player and describe their poses and environment.
            '''
            visual_description = self.llm.inference(prompt=visual_prompt)
            visual_image = generate_game_image(prompt=visual_description)
            visual_with_subtitle = self.add_llm_subtitle(image=visual_image, subtitle_text=f"[{speaker.name}]: {dialogue}")

            # Update objective status based on interaction and LLM response
            if self.check_objective_completed(quest):
                quest.update_progress(self.player.__dict__, self.llm)

            yield {"dialogue": dialogue, "image": visual_with_subtitle}

    def add_llm_subtitle(self, image, subtitle_text):
        """Add a subtitle to the generated image."""
        try:
            draw = ImageDraw.Draw(image)
            font = ImageFont.load_default()
            text_position = (20, image.height - 50)
            draw.text(text_position, subtitle_text, font=font, fill="white")
            return image
        except Exception as e:
            print(f"Failed to add subtitle: {e}")

    def get_speaker(self, dialogue_response, quest_npcs):
        """Identify the speaker based on LLM response."""
        speaker_name = dialogue_response[0]
        for npc in quest_npcs:
            if npc.name == speaker_name:
                return npc
        return {"name": "Narrator"}

    def check_objective_completed(self, quest):
        """Check if the quest's current objective is completed based on LLM feedback."""
        current_objective = quest.get_current_objective()
        quest_npcs = [npc for npc in self.npcs if npc.name in quest.npcs]

        prompt = f'''
        Act as a dungeon master checking the status of an RPG quest objective.
        Quest details: {quest.__str__()}.
        NPCs involved: {[npc.__str__() for npc in quest_npcs]}.
        Player details: {self.player.__str__()}.
        Current objective: {current_objective}.
        Reply only "yes" if the objective is fulfilled or "no" if not.
        '''
        response = self.llm.inference(prompt=prompt)
        return "yes" in response
