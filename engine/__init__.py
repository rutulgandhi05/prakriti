""" NPC package. """

from engine import scene
from engine import parse
from engine import step
from engine import prompt

__all__ = ["scene", "parse", "step", "prompt"]

from outlines import models
from transformers import AutoModelForCausalLM, AutoTokenizer

from engine.step import NPCStepper
from engine.quest import Quest
from engine.parse import CharacterAction
from engine.scene import Item, Character, Location, Skill, ProtagonistCharacter, NarratorCharacter, ParameterType

model_name = "Gigax/NPC-LLM-7B"
llm = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = models.Transformers(llm, tokenizer)


class Erin():
    def __init__(self, model):
        self.stepper = NPCStepper(model)

        self.events = []
    
    def remember_interaction(self, event: CharacterAction):
        self.events.append(event)

    
    def prompt(self, prompt):
        ch_prompt = CharacterAction(command="say", protagonist="Lyra", parameters= [prompt])
        self.remember_interaction(ch_prompt)

        events = self.events

        context, locations, NPCs, protagonist, items = self.context()
        
        res  = self.stepper.get_action(
            context=context,
            locations=locations,
            NPCs=NPCs,
            protagonist=protagonist,
            items=items,
            events=events,
        )

        if res:
            self.remember_interaction(res)

        return res

    def context(self):
        context="Old village"
        locations=[Location(name="Entrance of village", description="At the entrance of the village there is a gate which is a beautiful landmark."), 
                   Location(name="Dark Forest", description="There is a dense forest on the west outskirts of the village."), 
                   Location(name="Town House", description="In the middle of the village there is a town house. It is often used for public gatherings.")]
        NPCs=[Character(name="Lyra", description="A herbelist. Well versed with ayurveda.", current_location=Location(name="Dark Forest", description="There is a dense forest on the west outskirts of the village."))]
       
        protagonist=ProtagonistCharacter(
            name= "Erin",
            description="Gaurd of the village. Brave and skilled in combat.",
            current_location=Location(name="Entrance of village", description="At the entrance of the village there is a gate which is a beautiful landmark."),
            memories=[],
            quests= ["save the village from attackers and gaurd the main gate."],
            skills=[Skill(name="talk", description="Talk to anyone.", parameter_types=[ParameterType.character])]
        )
        
        
        items=[Item(name="sword", description="A big sharp blade")]
        self.events = [CharacterAction(command="walk", protagonist=protagonist.name, parameters=["towards gate"] )]
        
        return context, locations, NPCs, protagonist, items
       
    




class NARRATOR(NPCStepper):
    def __init__(self):
        super().__init__(model)

    def context(self):
        pass
    
    def prompt(self):

        context, locations, NPCs, protagonist, items, narrator, events = self.context()

        res  = self.get_narrator_update(
            context=context,
            locations=locations,
            NPCs=NPCs,
            protagonist=protagonist,
            narrator=narrator,
            items=items,
            events=events,
        )

        return res


class Master(NARRATOR):
    def context(self):
        context="Old village"
        
        locations=[Location("Entrance of village", "At the entrance of the village there is a gate which is a beautiful landmark."), 
                   Location("Dark Forest", "There is a dense forest on the west outskirts of the village."), 
                   Location("Town House", "In the middle of the village there is a town house. It is often used for public gatherings.")]
        
        NPCs=[Character("Lyra", "A herbelist. Well versed with ayurveda.", Location("Dark Forest", "There is a dense forest on the west outskirts of the village.")), 
              Character("Erin", "Gaurd of the village. Brave and skilled in combat.", Location("Entrance of village", "At the entrance of the village there is a gate which is a beautiful landmark."))]
       
        protagonist=ProtagonistCharacter(
            name= "Erin",
            description="Gaurd of the village. Brave and skilled in combat.",
            current_location="Entrance of the village.",
            memories=[],
            quests= [Quest("Save village", "save the village from attackers and gaurd the main gate.", "Village")],
            skills=[Skill(name="talk", description="Talk to anyone.", parameter_types=["character"])]
        )