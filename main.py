from engine.game_state_manager import GameStateManager
from engine.scene_step import SceneStep
from engine.dialogue_step import DialogueStep
from engine.llm_engine import LLMEngine

if __name__ == "__main__":
    game_state = {
        "current_scene": {
            "description": "A mystical forest.",
            "npc": "Eldara"
        }
    }

    llm_engine = LLMEngine()
    manager = GameStateManager(initial_state=game_state, llm_engine=llm_engine)
    manager.set_step(SceneStep(game_state, llm_engine))
    manager.run()