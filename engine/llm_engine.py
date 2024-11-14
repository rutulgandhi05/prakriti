from outlines import models, generate
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class LLMEngine:
    def __init__(self, model_name="Gigax/NPC-LLM-7B"):
        """ self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="auto"
        ) """
        self.model = models.transformers("microsoft/Phi-3-mini-4k-instruct", device="cuda")


    def generate_text(self, prompt, max_tokens=100):
        generator = generate.text(self.model)
        output = generator(prompt, max_tokens=max_tokens)
        return output

    def generate_scene_description(self, environment, mood, npc, event):
        prompt = (
            f"Create a scene: Environment={environment}, Mood={mood}, "
            f"NPC={npc}, Event={event}."
        )
        return self.generate_text(prompt)

    def generate_npc_response(self, npc, player_input, scene_description):
        prompt = (
            f"NPC '{npc}' in scene '{scene_description}' responds to "
            f"'{player_input}'."
        )
        return self.generate_text(prompt)


if __name__ == "__main__":
    llm = LLMEngine()
    print(llm.generate_scene_description("forest", "mystical", "Eldara", "artifact discovery"))
    print(llm.generate_npc_response("Eldara", "What brings me here?", "A mystical forest."))