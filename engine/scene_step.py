from engine.step import Step

class SceneStep(Step):
    def __init__(self, game_state, llm_engine):
        super().__init__(game_state)
        self.llm_engine = llm_engine

    def execute(self):
        environment = "forest"
        mood = "mystical"
        npc = "Eldara"
        event = "artifact discovery"

        scene_description = self.llm_engine.generate_scene_description(environment, mood, npc, event)
        self.game_state["current_scene"] = scene_description
        print(f"Scene: {scene_description}")

    def next_step(self):
        return "DialogueStep"
