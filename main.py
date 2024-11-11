import logging

from dialogue_manager import DialogueManager
from db_manager import DBManager
from image_manager import ImageManager

logging.basicConfig(
    filename="game_logs.txt",
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

def display_scene(scene_id, description, image_path, npcs):
    logging.info(f"Displaying scene {scene_id}")

    display_text = []
    
    display_text.append(f"\nScene {scene_id}: {description}")
    if npcs:
        display_text.append("NPCs Present:"+ ", ".join(npcs))
    if image_path:
        try:
            display_text.append(image_path)
        except Exception as e:
            display_text.append("Could not display image:", e)
    else:
        display_text.append("No image available for this scene.")

    logging.info(''.join(display_text).replace(',', ' '))

def main():
    simulated_inputs = [
        {"type": "action", "value": "explore the forest"},
        {"type": "scene_choice", "value": 2},
        {"type": "action", "value": "talk to the old traveler"},
        {"type": "scene_choice", "value": 4},
        {"type": "action", "value": "inspect the ruins"},
    ]

    logging.info("Starting main game loop")

    # Initialize components
    dbmanager = DBManager()
    dialogue_manager = DialogueManager()
    image_manager = ImageManager(save_directory="generated_images")

    
    # Set initial scene ID and simulated input index
    current_scene_id = 1
    input_index = 0

    while input_index < 5:
        logging.info("Intro")
        record = dbmanager.get_scene_by_id(current_scene_id)
        description, npcs = record["description"], record["npcs"]

        if not description:
            logging.error(f"Scene ID {current_scene_id} not found, ending game.")
            break

        enhanced_description = dialogue_manager.generate_enhanced_description(description)
        image_path = image_manager.generate_image(enhanced_description, current_scene_id)
       
        # Simulated input for player action or scene choice
        simulated_input = simulated_inputs[input_index]
        input_index += 1
        
        if simulated_input["type"] == "action":
            # Process player action
            logging.info(f"\n Player: {simulated_input['value']}")
            npc_response = dialogue_manager.handle_dialogue(simulated_input['value'], current_scene_id)
            logging.info(f"\nNPC: {npc_response} ")

        elif simulated_input["type"] == "scene_choice":
            # Retrieve next scene options and proceed with choice
            next_scene_options = dbmanager.get_next_scene_options(current_scene_id)
            if next_scene_options:
                logging.info("\nAvailable Paths:")
                for option in next_scene_options:
                    logging.info(f"{option['id']}: {option['description']}")
                
                selected_scene = simulated_input["value"]
                logging.info(f"\n Player: {simulated_input['value']}" )
                if any(option['id'] == selected_scene for option in next_scene_options):
                    current_scene_id = selected_scene
                    logging.info(f"Proceeding to Scene {selected_scene} based on simulated input.")
                else:
                    logging.info("Invalid simulated scene selection. Ending test.")
                    break
            else:
                logging.info("\nNo further paths from this scene. The adventure ends here.")
                break

    # Close resources
    dbmanager.close()
    dialogue_manager.close()

    logging.info("Automated adventure test completed. Goodbye!")

if __name__ == "__main__":
    main()
