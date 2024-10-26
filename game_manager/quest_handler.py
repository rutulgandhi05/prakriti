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
        :param npcs: List of NPCs involved in the quest.
        """
        self.quest_id = quest_id
        self.name = name
        self.description = description
        self.objectives = objectives
        self.current_objective_index = 0
        self.status = "not_started"
        self.npcs = npcs

    def start(self):
        """Marks the quest as started and updates its status."""
        self.status = "in_progress"
        self.add_to_db()

    def check_objective_completion(self, current_objective, game_state):
        """
        Checks if the current objective is complete based on the game state.
        :param current_objective: The current objective dictionary.
        :param game_state: A dictionary representing the player's state (e.g., items, actions, location).
        :return: Boolean indicating whether the objective is complete.
        """
        # Placeholder logic to check if an objective's condition is met in the game state
        # You could check for specific items, locations, or interactions here
        return game_state.get(current_objective.get('condition'), False)

    def update_progress(self, game_state, llm):
        """
        Updates the quest's progress by checking if the current objective condition is met.
        Uses the LLM to generate dynamic story responses when objectives are completed.
        :param game_state: A dictionary representing the player's state.
        :param llm: The LLM object used to generate dynamic responses.
        """
        if self.status != "in_progress":
            print(f"Quest '{self.name}' is not active.")
            return
        
        current_objective = self.objectives[self.current_objective_index]

        # Check if the condition of the current objective is fulfilled
        if self.check_objective_completion(current_objective, game_state):
            # Move to the next objective
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
                
                print(f"New Objective: {next_objective}\nLLM Response: {llm_response}")
                
        self.add_to_db()

    def complete(self, llm):
        """
        Marks the quest as completed and generates a final LLM response.
        :param llm: The LLM object to generate a final response for the completed quest.
        """
        self.status = "completed"
        final_response = llm.generate_response(
            quest_name=self.name,
            player_action="quest completed",
            player_state={},
            quest_background=self.description
        )
        
        print(f"Quest '{self.name}' completed!\nLLM Final Response: {final_response}")
        self.add_to_db()

    def add_to_db(self):
        """Saves the current quest state to the database."""
        db = DatabaseManager()
        query = '''
            INSERT OR REPLACE INTO quest (id, name, description, objectives, status, objective_index)
            VALUES (?, ?, ?, ?, ?, ?);
        '''
        data = (
            self.quest_id,
            self.name,
            self.description,
            json.dumps(self.objectives),
            self.status,
            self.current_objective_index
        )
        db.cursor.execute(query, data)
        db.conn.commit()

    def get_status(self):
        """Returns the current status of the quest."""
        return self.status

    def get_current_objective(self):
        """Returns the current objective of the quest."""
        if self.status == "in_progress" and self.objectives:
            return self.objectives[self.current_objective_index]
        return None

    def __str__(self):
        """
        String representation of the quest object for easy inspection.
        :return: A string with quest details.
        """
        return (f"Quest: {self.name}, Description: {self.description}, "
                f"Current Objective: {self.get_current_objective()}, Status: {self.status}, "
                f"NPCs Involved: {', '.join(self.npcs)}")
