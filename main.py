# main.py

from dialogue_manager import DialogueManager
from db_manager import DBManager
from image_manager import ImageManager

# Simulated inputs to automate gameplay for testing
# You can adjust the actions and scene choices as needed
simulated_inputs = [
    {"type": "action", "value": "explore the forest"},
    {"type": "scene_choice", "value": 2},
    {"type": "action", "value": "talk to the old traveler"},
    {"type": "scene_choice", "value": 4},
    {"type": "action", "value": "inspect the ruins"},
]

def display_scene(scene_id, description, image_path, npcs):
    """Display the current scene description, NPCs, and image."""
    print(f"\nScene {scene_id}: {description}")
    if npcs:
        print("NPCs Present:", ", ".join(npcs))
    if image_path:
        try:
            print(image_path)
        except Exception as e:
            print("Could not display image:", e)
    else:
        print("No image available for this scene.")

def main():
    # Initialize components
    db_manager = DBManager()
    dialogue_manager = DialogueManager()
    image_manager = ImageManager(save_directory="generated_images")

    print("Starting the automated adventure test...\n")

    # Set initial scene ID and simulated input index
    current_scene_id = 1
    input_index = 0

    while input_index < len(simulated_inputs):
        # Retrieve and display the current scene details
        description, npcs = db_manager.get_scene_by_id(current_scene_id)
        enhanced_description = dialogue_manager.generate_enhanced_description(description)
        image_path = image_manager.generate_image(enhanced_description, current_scene_id)
        
        display_scene(current_scene_id, enhanced_description, image_path, npcs)

        # Simulated input for player action or scene choice
        simulated_input = simulated_inputs[input_index]
        input_index += 1

        if simulated_input["type"] == "action":
            # Process player action
            npc_response = dialogue_manager.handle_dialogue(simulated_input["value"])
            print("\nNPC:", npc_response)

        elif simulated_input["type"] == "scene_choice":
            # Retrieve next scene options and proceed with choice
            next_scene_options = db_manager.get_next_scene_options(current_scene_id)
            if next_scene_options:
                print("\nAvailable Paths:")
                for option in next_scene_options:
                    print(f"{option['id']}: {option['description']}")
                
                selected_scene = simulated_input["value"]
                if any(option['id'] == selected_scene for option in next_scene_options):
                    current_scene_id = selected_scene
                    print(f"Proceeding to Scene {selected_scene} based on simulated input.")
                else:
                    print("Invalid simulated scene selection. Ending test.")
                    break
            else:
                print("\nNo further paths from this scene. The adventure ends here.")
                break

    # Close resources
    db_manager.close()
    dialogue_manager.close()
    print("Automated adventure test completed. Goodbye!")

if __name__ == "__main__":
    main()
