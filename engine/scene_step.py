from engine.step import Step
import random

class SceneStep(Step):
    def __init__(self, game_state, llm_engine):
        super().__init__(game_state)
        self.llm_engine = llm_engine

    def execute(self):
        """
        Generate a new scene using the LLM and update the game state.
        """
        environment, mood, npc, event = self.generate_scene_components()
        scene_data = self.llm_engine.generate_scene_description(environment, mood, npc, event)

        if "error" in scene_data:
            print("Error generating scene:", scene_data["error"])
            print("Raw response:", scene_data["raw_response"])
            return

        self.game_state["current_scene"] = scene_data
        print("\nScene:")
        print(scene_data["description"])


    def next_step(self):
        """
        Transition to the DialogueStep.
        """
        return "DialogueStep"

    def generate_scene_components(self):
        """
        Provide random components for the scene.
        """
        import random
        environments = ["forest", "mountain", "riverbank", "cave", "ruins"]
        moods = ["eerie", "calm", "mystical", "desolate"]
        npcs = ["a trader", "a guardian", "a hermit", "a traveler"]
        events = ["a hidden path is revealed", "strange sounds echo", "an artifact is found"]

        return random.choice(environments), random.choice(moods), random.choice(npcs), random.choice(events)