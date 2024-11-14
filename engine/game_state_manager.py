from engine.dialogue_step import DialogueStep
from llm_engine import LLMEngine

class GameStateManager:
    def __init__(self, initial_state, llm_engine):
        self.state = initial_state
        self.current_step = None
        self.llm_engine = llm_engine

    def set_step(self, step):
        self.current_step = step

    def run(self):
        while self.current_step:
            self.current_step.execute()
            next_step_name = self.current_step.next_step()
            if next_step_name:
                step_class = globals()[next_step_name]
                # Pass the required llm_engine when initializing the next step
                if next_step_name == "DialogueStep":
                    print(" state", self.state)
                    print(type(self.state))
                    self.current_step = step_class(self.state, self.llm_engine)
                elif next_step_name == "SceneStep":
                    self.current_step = step_class(self.state, self.llm_engine)
                else:
                    self.current_step = step_class(self.state, self.llm_engine)
            else:
                self.current_step = None
