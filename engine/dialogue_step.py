from engine.step import Step

class DialogueStep(Step):
    def __init__(self, game_state, llm_engine):
        super().__init__(game_state)
        self.llm_engine = llm_engine

    def execute(self):
        npc = self.game_state["current_scene"]["npc"]
        player_input = "What brings me here?"
        npc_response = self.llm_engine.generate_npc_response(npc, player_input, self.game_state["current_scene"]["description"])
        print(f"{npc}: {npc_response}")

    def next_step(self):
        return "SceneStep"
