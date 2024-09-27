from npc_handler import NPC
from quest_handler import Quest

class QuestBlackmoorCurse(Quest):
    def __init__(self, story_manager):
        super().__init__(
            quest_id=1,
            name="The Blackmoor Curse",
            description="Investigate the mysterious curse gripping Blackmoor.",
            objectives=[
                {"id": 1, "description": "Investigate the town", "completed": False},
                {"id": 2, "description": "Explore Blackmoor Forest", "completed": False},
                {"id": 3, "description": "Confront the Witch", "completed": False}
            ]
        )
        self.npcs = [
            NPC(id=1, name="Captain Merrick", profession="Guard Captain", temperament="Brave", backstory="Captain of Blackmoor's guards."),
            NPC(id=2, name="Lira", profession="Herbalist", temperament="Reclusive", backstory="An herbalist with knowledge of the curse."),
            NPC(id=3, name="Old Man Calen", profession="Madman", temperament="Unstable", backstory="Claims to have seen the witch and survived.")
        ]
        self.story_manager = story_manager
        self.witch_confronted = False

    def confront_witch(self):
        print("You encounter the Cursed Witch in the forest...")
        choice = input("Do you want to attack the witch or investigate further? (attack/investigate): ").lower()

        if choice == "attack":
            print("You chose to attack the witch!")
            self.witch_confronted = True
            # Track this in the player's story progress
            self.story_manager.update_story_progress('witch_attacked', True)
            # Add the interaction to NPC memory (if the witch is an NPC or tied to one)
        else:
            print("You chose to investigate further.")
            # Track this in the player's story progress
            self.story_manager.update_story_progress('witch_attacked', False)
            # Add the interaction to NPC memory (if applicable)

        # Save the choice in memory for future reference
        for npc in self.npcs:
            npc.remember_interaction(f"Player choice: {choice}", "Witch interaction")
        
        self.complete_quest()

    def complete_quest(self):
        print("You have completed the quest: The Blackmoor Curse.")
        if self.witch_confronted:
            print("Killing the witch will have consequences later.")
        else:
            print("You uncovered the truth about the curse.")
        self.story_manager.complete_quest(self)
