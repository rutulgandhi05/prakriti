import json
from db_master.db_manager import DatabaseManager

class Quest:
    def __init__(self, quest_id, name, description, objectives):
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
        self.status = "not_started"  # Quest status: "not_started", "in_progress", "completed"
        self.player_progress = {} 


    def start(self):
        """
        Marks the quest as started and updates its status.
        """
        self.status = "in_progress"
        self.add_to_db()
        print(f"Quest '{self.name}' started!")


    def update_progress(self, game_state):
        """
        Updates the quest's progress by checking if the current objective condition is met.
        :param game_state: A dictionary representing the player's state (e.g., knowledge, items, location).
        """
        if self.status != "in_progress":
            print(f"Quest '{self.name}' is not active.")
            return
        
        current_objective = self.objectives[self.current_objective_index]

        # Check if the condition of the current objective is fulfilled
        if self.check_objective_completion(current_objective, game_state):
            print(f"Objective '{current_objective['description']}' completed!")
            self.current_objective_index += 1

            if self.current_objective_index >= len(self.objectives):
                self.complete()
            else:
                print(f"Next objective: {self.objectives[self.current_objective_index]['description']}")
        else:
            print(f"Current objective not completed yet. Objective: {current_objective['description']}")

        self.add_to_db()


    def check_objective_completion(self, objective, game_state):
        """
        Check if the objective's condition is satisfied by the player's game state.
        :param objective: The current objective to check.
        :param game_state: The player's current state (e.g., knowledge, items, etc.).
        :return: True if the objective is completed, False otherwise.
        """
        # The objective's condition could be any dynamic check on the player's state
        condition = objective['condition']
        return condition(game_state)  # Call the condition function with the game state


    def complete(self):
        """
        Marks the quest as completed and updates its status.
        """
        self.status = "completed"
        print(f"Quest '{self.name}' completed!")
        self.add_to_db()


    def add_to_db(self):
        db = DatabaseManager()

        query = '''
            INSERT OR REPLACE INTO quest (name, description, objectives, status)
            VALUES (?, ?, ?, ?);
        '''
        data = (self.name, self.description, json.dumps(self.objectives), self.status)
        
        db.cursor.execute(query, data)


    def get_status(self):
        """Returns the current status of the quest."""
        return self.status
    
    def get_current_objective(self):
        """Returns the current objective of the quest."""
        if self.status == "in_progress":
            return self.objectives[self.current_objective_index]
        return None 
        
