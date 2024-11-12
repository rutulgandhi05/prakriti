import logging
from outlines import models
from llama_cpp import Llama
from engine.step import NPCStepper
from engine.parse import CharacterAction
from engine.scene import Character, Item, Location, ProtagonistCharacter, Skill, ParameterType




logging.basicConfig(
    filename="game_logs.txt",
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

def main():
    llm = Llama.from_pretrained(
        repo_id="Gigax/NPC-LLM-3_8B-GGUF",
        filename="npc-llm-3_8B.gguf",
        # n_gpu_layers=-1, 
        n_ctx=2048, 
        )
    
 
    model = models.LlamaCpp(llm) 

    stepper = NPCStepper(model=model)


    context = "Medieval world"
    current_location = Location(name="Old Town", description="A quiet and peaceful town.")
    locations = [current_location] # you can add more locations to the scene
    NPCs = [
        Character(
        name="John the Brave",
        description="A fearless warrior",
        current_location=current_location,
        )
    ]
    protagonist = ProtagonistCharacter(
        name="Aldren",
        description="Brave and curious",
        current_location=current_location,
        memories=["Saved the village", "Lost a friend"],
        quests=["Find the ancient artifact", "Defeat the evil warlock"],
        skills=[
            Skill(
                name="Attack",
                description="Deliver a powerful blow",
                parameter_types=[ParameterType.character],
            )
        ],
        psychological_profile="Determined and compassionate",
    )
    items = [Item(name="Sword", description="A sharp blade")]
    events = [
        CharacterAction(
            command="Say",
            protagonist=protagonist,
            parameters=[items[0], "What a fine sword!"],
        )
    ]

    action = stepper.get_action(
        context=context,
        locations=locations,
        NPCs=NPCs,
        protagonist=protagonist,
        items=items,
        events=events,
    )
    
    logging.info(f"Aldren: {action}")

if __name__ == "__main__":
    main()
