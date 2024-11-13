from engine.step import Step
import random

class SceneStep(Step):
    """
    Handles the logic for generating and managing dynamic scenes.
    """

    def execute(self):
        """
        Generate a new scene and update the game state.
        """
        environment, mood, npc, event = self.generate_scene_components()
        self.game_state["current_scene"] = {
            "description": f"A {mood} {environment} where {event}. You encounter {npc}.",
            "npc": npc,
        }
        print("\nScene:")
        print(self.game_state["current_scene"]["description"])

    def next_step(self):
        """
        Transition to the DialogueStep.
        """
        return "DialogueStep"

    def generate_scene_components(self):
        """
        Generate random components for the scene.
        """
        environments = ["forest", "mountain", "riverbank", "cave", "ruins"]
        moods = ["eerie", "calm", "mystical", "desolate"]
        npcs = ["a trader", "a guardian", "a hermit", "a traveler"]
        events = ["a hidden path is revealed", "strange sounds echo", "an artifact is found"]

        return random.choice(environments), random.choice(moods), random.choice(npcs), random.choice(events)
