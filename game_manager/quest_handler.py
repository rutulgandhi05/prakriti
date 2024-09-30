import json
from db_master.db_manager import DatabaseManager
from thechosenone.inference import generate_game_image


class Quest:
    def __init__(self, quest_id, name, description, objectives, npcs):
        """
        Initializes a Quest with basic attributes and a list of objectives.
        :param quest_id: Unique identifier for the quest.
        :param name: The name of the quest.
        :param description: A brief description of the quest.
        :param objectives: A list of objectives the player must complete.
        """
        self.quest_id = quest_id
        self.name = name
        self.description = description
        self.objectives = objectives  # List of quest objectives
        self.current_objective_index = 0  # Tracks the current objective
        self.quest_summary = ""
        self.recent_events = []
        self.status = "not_started"  # Quest status: "not_started", "in_progress", "completed"
        self.npcs = npcs


    def start(self):
        """
        Marks the quest as started and updates its status.
        """
        self.status = "in_progress"
        self.add_to_db()


    def update_progress(self, game_state, llm):
        """
        Updates the quest's progress by checking if the current objective condition is met.
        Uses the LLM to generate dynamic story responses.
        :param game_state: A dictionary representing the player's state (e.g., knowledge, items, location).
        :param llm: The LLM object used to generate dynamic responses.
        """
        if self.status != "in_progress":
            print(f"Quest '{self.name}' is not active.")
            return
        
        current_objective = self.objectives[self.current_objective_index]

        # Check if the condition of the current objective is fulfilled
        if self.check_objective_completion(current_objective, game_state):
            self.current_objective_index += 1

            if self.current_objective_index >= len(self.objectives):
                self.complete(llm)
            else:
                next_objective = self.objectives[self.current_objective_index]['description']

                # Generate LLM response for the new objective
                llm_response = llm.generate_response(
                    quest_name=self.name,
                    player_action=f"completed {current_objective['description']}",
                    player_state=game_state,
                    quest_background=self.description
                )
                
        self.add_to_db()



    def complete(self, llm):
        """
        Marks the quest as completed and generates a final LLM response.
        :param llm: The LLM object to generate a final response for the completed quest.
        """
        self.status = "completed"        
        self.add_to_db()


    def add_to_db(self):
        db = DatabaseManager()

        query = '''
            INSERT OR REPLACE INTO quest (name, description, objectives, status)
            VALUES (?, ?, ?, ?);
        '''
        data = (self.name, self.description, json.dumps(self.objectives), self.status)
        
        db.cursor.execute(query, data)
        db.conn.commit()
    

    def get_status(self):
        """Returns the current status of the quest."""
        return self.status
    
    def get_current_objective(self):
        """Returns the current objective of the quest."""
        if self.status == "in_progress":
            if self.objectives is []:
                self.complete()
                return None
            return self.objectives[self.current_objective_index]
        return None 
        

    def __str__(self):
        """
        String representation of the player object for easy inspection.
        :return: A string with player details
        """
        return f"Quest: {self.name}, Description: {self.description}, Objectives: {self.get_current_objective()}, Npcs required: {self.npcs}, current_objective: {self.get_current_objective()}"