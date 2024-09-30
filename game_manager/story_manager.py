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
        self.player = None  # Placeholder for player character
        self.npcs = []  # List to hold NPC objects
        self.quests = []  # List to hold quests
        self.player = None


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
        name = player_data.get('name', 'Unknown Hero')
        backstory = player_data.get('backstory', 'A mysterious figure...')
        profession = player_data.get('profession', 'A mysterious figure...')
        current_state = player_data.get('current_state', 'A mysterious figure...')
        inventory = player_data.get('inventory', 'A mysterious figure...')
        self.player = Player(name=name, backstory=backstory, profession=profession, current_state=current_state, inventory=inventory)
        Player.add_to_db()


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
            npc.add_to_db()  # Store NPC in the database
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
            
                # Run the quest objectives
            self.run_quest(quest)

    
    def run_quest(self, quest):
        """
        Handle quest objectives, including player actions, LLM responses, and visual progression.
        """
        while quest.status == "in_progress":

            current_objective = quest.get_current_objective()
            
            if not current_objective:
                print("No more objectives to complete.")
                quest.complete()
                break
            
            quest_npcs = [npc for npc in self.npcs if npc.name in quest.npcs]
            quest_details = quest.__str__()
            player_details = self.player.__str__()
            npc_details = [npc_details.append(npc.__str_())  for npc in quest_npcs]
            

            dialouge_prompt = f'''Act as a dungeoun master. You are running a RPG adventure quest. It is played by the player.The player details are [{player_details}].
            The quest deatils are [{quest_details}].
            The npcs required and for this quests and their details are: [{npc_details}].
            Current objective of the quest is : {current_objective}.
            If player choose to interact with enviorment,  narrator is the one who speaks next and will describe the scene after player action.
            You can only choose between narrator and npcs for next to speak. You have to act as the character and think accordingly to the character characteristics.
            Based on the current state, determine who speaks next. Generate the response, including any emotional tones or gestures that fit the situation.
            based on the memory of the npc determine its relationship level with player.
            Give me the output in the form of dialouges: ["speaker", "speakers dialouge"].
            '''

            dialouge_responses = list(self.llm.inference(prompt=dialouge_prompt))

            speaker = self.get_speaker(dialouge_responses)
            dialouge = dialouge_responses[1]

            interaction  = {"player": self.player.current_move, speaker.name: dialouge}

            if speaker.name != "Narrator":
                speaker.remember_interaction(interaction)


            visual_propmt = f'''Act as a professional prompt engineer and write a prompt that can be passed to sdxl model to create visual of current scene of the quest based on the information provided.
            The quest deatils are [{quest_details}].
            The npcs required and for this quests and their details are: [{npc_details}].
            Current objective of the quest is : {current_objective}.
            After the player did {self.player.current_move}, the {speaker.name} said {dialouge}.
            If the player choose to interact with anything other than npcs just create a picture of player and the enviourment around it. 
            Only show npcs who player interact with or see.
            Give me the prompt in such a way that it describe poses and location of each and every entity in the scene.
            '''

            gen_visual_propmt = self.llm.inference(prompt=visual_propmt)

            viusal = generate_game_image(prompt=gen_visual_propmt)
            visual_sub = self.add_llm_subtitle(image=viusal, subtitle_text=f"[{speaker.name}]: {dialouge}")

            if self.check_objective_completed(quest=quest):
                quest.objectives = quest.objectives[quest.current_objective_index+1 :]

            yield {"dialouge": dialouge_responses, "image": visual_sub}


    
    def add_llm_subtitle(self, image, subtitle_text):
        """Add a subtitle (LLM response) to the generated image."""
        try:
            draw = ImageDraw.Draw(image)

            # Define font and text position
            font = ImageFont.load_default()  # You can load a custom font if needed
            image_width, image_height = image.size
            text_position = (20, image_height - 50)  # Position the subtitle near the bottom

            # Add text to the image
            draw.text(text_position, subtitle_text, font=font, fill="white")

            return image
            # Save the modified image
            # image.save(image_path)

        except Exception as e:
            print(f"Failed to add subtitle: {e}")


    def get_speaker(self, dialouge_responses, quest_npcs):
        for npc in quest_npcs:
            if npc.name == dialouge_responses[0]:
                return npc
        else:
            return {"name":  "Narrator"}
        

    def check_objective_completed(self, quest):
        quest_npcs = [npc for npc in self.npcs if npc.name in quest.npcs]
        quest_details = quest.__str__()
        player_details = self.player.__str__()
        npc_details = [npc_details.append(npc.__str_())  for npc in quest_npcs]

        prompt = f'''Act as a dungeoun master. You are running a RPG adventure quest. It is played by the player.The player details are [{player_details}].
            The quest deatils are [{quest_details}].
            The npcs required and for this quests and their details are: [{npc_details}].
            Current objective of the quest is : {quest.get_current_objective()}.
            Check if current objective if sullfilled by the player. Reply in only yes or no.
           '''

        response = self.llm.inference(prompt=prompt)

        if "yes" in response:
            return True
        else:
            return False