from engine.step import Step

class DialogueStep(Step):
    """
    Handles the logic for NPC interactions.
    """

    def execute(self):
        """
        Generate NPC dialogue based on the current scene.
        """
        npc = self.game_state["current_scene"]["npc"]
        print(f"\n{npc.capitalize()} says: 'Welcome traveler, what brings you here?'")
        player_input = input("Your response: ")
        self.game_state["last_input"] = player_input

    def next_step(self):
        """
        Return to SceneStep or end the game.
        """
        return "SceneStep" if input("Continue? (y/n): ").lower() == "y" else None
