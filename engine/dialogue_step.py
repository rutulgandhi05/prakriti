from engine.step import Step

class DialogueStep(Step):
    def __init__(self, game_state, llm_engine):
        super().__init__(game_state)
        self.llm_engine = llm_engine

    def execute(self):
        """
        Generate NPC dialogue using the LLM based on the current scene.
        """
        npc = self.game_state["current_scene"]["npc"]
        scene_description = self.game_state["current_scene"]["description"]

        player_input = input(f"\n{npc.capitalize()} says: 'What brings you here?' Your response: ")

        npc_response = self.llm_engine.generate_npc_response(player_input, scene_description, npc)

        if "error" in npc_response:
            print("Error generating NPC response:", npc_response["error"])
            print("Raw response:", npc_response["raw_response"])
            return

        print(f"{npc.capitalize()} responds: {npc_response['npc_response']}")


    def next_step(self):
        return "SceneStep" if input("Continue? (y/n): ").lower() == "y" else None