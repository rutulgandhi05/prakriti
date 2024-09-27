import yaml
from llm_master.llm import LLM
from npc_handler import NPC
from quest_handler import Quest
from db_master.db_manager import DatabaseManager
from thechosenone.inference import generate_game_image
from PIL import Image, ImageDraw, ImageFont 

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
        self.player = {'name': player_name, 'backstory': player_backstory}
        print(f"Welcome, {player_name}! {player_backstory}")


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
        print(intro_text)


    def play(self):
        """
        Main gameplay loop with default quest sequence.
        """
        self.setup_game()  # Ensure NPCs, Quests, and Player are set up
        self.start_intro()  # Always start with the story intro

        # Default quest sequence
        print("Starting Quest 1: The Blackmoor Curse")
        self.play_quest(1)

        print("Starting Quest 2: The Mayor’s Secret")
        self.play_quest(2)

        print("Starting Quest 3: The Demon’s Reckoning")
        self.play_quest(3)

        print("All quests completed! The adventure has ended.")

    
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
            
            print(f"Playing Quest: {quest.name}")
            
            # Generate visuals for the current stage of the quest
            self.generate_quest_visual(quest)

            # Run the quest objectives
            self.run_quest_logic(quest)

            # After all objectives are completed, mark the quest as complete
            if quest.status == "completed":
                print(f"Quest '{quest.name}' completed!")
            else:
                print(f"Quest '{quest.name}' is still in progress.")
        else:
            print("Quest not found.")

    
    def run_quest_logic(self, quest):
        """
        Handle quest objectives, including player actions, LLM responses, and visual progression.
        """
        # Simulate player's game state (placeholder, could be based on actual player input)
        game_state = {
            "has_item": False,  # Example condition
            "visited_location": False,
            "talked_to_npc": False,
        }

        while quest.status == "in_progress":
            current_objective = quest.get_current_objective()
            
            if not current_objective:
                print("No more objectives to complete.")
                quest.complete()
                break
            
            print(f"Current Objective: {current_objective['description']}")
            
            # Simulate player fulfilling the objective condition
            if self.fulfill_objective(current_objective, game_state):
                print(f"Objective '{current_objective['description']}' completed!")
                
                # Generate LLM response for the scene
                npc_context = self.llm.generate_response(quest.name, current_objective['description'])
                print(f"LLM Response: {npc_context}")  # Display the LLM response as part of the narrative
                
                # Update quest progress
                quest.update_progress(game_state)
                
                # Generate a new visual based on LLM response and current quest state
                self.generate_quest_visual(quest, llm_response=npc_context)
            else:
                print(f"Objective '{current_objective['description']}' not completed. Please try again.")
                break  # If the objective is not complete, stop and wait for further player actions


    def fulfill_objective(self, objective, game_state):
        """
        Check whether the player's game state fulfills the objective's conditions.
        :param objective: The current quest objective to check.
        :param game_state: The current state of the game (e.g., player's actions, items, locations).
        :return: True if the objective is fulfilled, False otherwise.
        """
        condition = objective.get("condition", None)
        
        # Here, we simulate fulfilling the objective. In a real game, this would depend on actual game events.
        if condition == "find_item":
            game_state["has_item"] = True
        elif condition == "visit_location":
            game_state["visited_location"] = True
        elif condition == "talk_to_npc":
            game_state["talked_to_npc"] = True
        
        # Check if the condition is fulfilled
        return game_state.get(condition, False)
    

    def generate_quest_visual(self, quest, llm_response=""):
        """Generate and display visuals for a quest using the Stable Diffusion model."""
        prompt = self.get_quest_prompt(quest)
        character_name = self.get_quest_character(quest)
        
        # Call the generate_game_image function to create the visual
        image = generate_game_image(character_name=character_name, prompt_postfix=prompt)
        image = self.add_llm_subtitle(image, llm_response)
        return image


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

        

    def get_quest_prompt(self, quest):
        """Generate a prompt for the current quest, including NPC poses and environmental context."""
        if quest.name == "The Blackmoor Curse":
            return "standing in the dark forest, surrounded by creatures"

        elif quest.name == "The Mayor’s Secret":
            return "in the mayor's office at night, standing behind his desk"

        elif quest.name == "The Demon’s Reckoning":
            return "in the ruined temple, preparing for battle with the demon"

        # Default prompt in case no specific quest context is found
        return f"in the scene of the quest '{quest.name}'"


    def get_quest_character(self, quest):
        """Return the primary character for the quest."""
        if quest.name == "The Blackmoor Curse":
            return "Captain Merrick"
        elif quest.name == "The Mayor’s Secret":
            return "Mayor Balthas"
        elif quest.name == "The Demon’s Reckoning":
            return "The Demon"
        
        return "Unnamed Character"              