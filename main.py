# main.py

from engine.game_state_manager import GameStateManager
from engine.scene_step import SceneStep
from engine.dialogue_step import DialogueStep

if __name__ == "__main__":
    # Initial game state
    game_state = {}

    # Create the state manager
    manager = GameStateManager(initial_state=game_state)

    # Start with the SceneStep
    manager.set_step(SceneStep(game_state))
    manager.run()

    print("\nGame Over!")
