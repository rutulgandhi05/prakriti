class Step:
    """
    Base class for all steps in the game.
    """

    def __init__(self, game_state):
        """
        Initialize the step with the shared game state.
        Args:
            game_state (dict): A dictionary tracking the overall game state.
        """
        self.game_state = game_state

    def execute(self):
        """
        Executes the logic for this step.
        Must be overridden in subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def next_step(self):
        """
        Determines the next step in the game.
        Must be overridden in subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method.")
