# main.py
from dialogue_manager import DialogueManager

def main():
    # Initialize dialogue manager
    dialogue_manager = DialogueManager()
    
    print("Welcome to the interactive narrative game!")
    print("Type your actions or questions, and interact with NPCs to advance the story.")
    print("Type 'exit' to end the game.\n")

    while True:
        # Display current scene context
        print(f"\nCurrent Scene: {dialogue_manager.story_engine.current_context}")
        
        # Display next scene options
        next_scenes = dialogue_manager.get_next_scenes()
        if next_scenes:
            print("\nPossible next scenes:")
            for option in next_scenes:
                print(f"- {option['id']}: {option['context']}")

        # Get player input
        player_input = input("\nYou: ")
        if player_input.lower() == "exit":
            print("Thanks for playing!")
            break

        # Process player input and get NPC response
        npc_response = dialogue_manager.handle_dialogue(player_input)
        print("NPC:", npc_response)
        
        # Ask player for scene transition if next scenes are available
        if next_scenes:
            next_scene_id = input("Choose the next scene by ID or press Enter to stay: ").strip()
            if next_scene_id.isdigit():
                dialogue_manager.change_scene(int(next_scene_id))
    
    # Close the database connection
    dialogue_manager.story_engine.db.close()

if __name__ == "__main__":
    main()
