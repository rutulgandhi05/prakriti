import logging
import torch
from outlines import models
from engine.step import NPCStepper
from engine.parse import CharacterAction
from engine.scene import Character, Item, Location, ProtagonistCharacter, Skill, ParameterType

from transformers import AutoModelForCausalLM, AutoTokenizer


logging.basicConfig(
    filename="game_logs.txt",
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

def main():
    model_name = "Gigax/NPC-LLM-7B"
    llm = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
 
    model = models.Transformers(llm, tokenizer)

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
            parameters=[str(items[0]), "What a fine sword!"],
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

    del model
    del tokenizer
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
