class GameStateManager:
    """
    Manages the game's state and controls the flow between steps.
    """

    def __init__(self, initial_state):
        self.state = initial_state
        self.current_step = None
        print("Available global classes:", globals().keys())


    def set_step(self, step):
        """
        Set the current step in the game.
        Args:
            step (Step): The step to execute.
        """
        self.current_step = step

    def run(self):
        """
        Execute the current step and transition to the next one.
        """
        while self.current_step:
            self.current_step.execute()
            next_step_class_name = self.current_step.next_step()
            if not next_step_class_name:
                break
            step_class = globals()[next_step_class_name]
            self.current_step = step_class(self.state)
