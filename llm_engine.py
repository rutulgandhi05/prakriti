import torch
import logging

from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login

login(token="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")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LLMEngine:
    def __init__(self, model_name="mistralai/Mistral-7B-v0.1"):
        logging.info("Initializing LLMEngine")
        self.model_name = model_name

        self.base_theme = "Ancient Fantasy Forest, mystical atmosphere, glowing lights, twilight colors"

    def generate_response(self, player_input):
        # Load the model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16
        ).to('cuda')
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Tokenize the input prompt
        inputs = tokenizer(player_input, return_tensors="pt").to('cuda')

        # Generate the response
        outputs = model.generate(
            **inputs,
            max_new_tokens=500,  # Limits the number of new tokens generated
            temperature=0.7,
            do_sample=True,      # Enables sampling for more diverse outputs
            top_p=0.9,           # Applies nucleus sampling
            top_k=50             # Limits the sampling pool to top 50 tokens
        )

        # Extract the generated tokens (excluding the input prompt)
        generated_tokens = outputs[0, inputs['input_ids'].shape[1]:]

        # Decode the generated tokens
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        # Clean up
        del model
        del tokenizer
        torch.cuda.empty_cache()

        return response
    

    def generate_npc_prompt(self, scene_description, npc_name, player_input):
        prompt = (
            f"Scene description: {scene_description}. "
            f"NPC present: {npc_name}. "
            f"Player says: {player_input}. "
            f"Respond as {npc_name} based on the scene and player input."
        )
        return prompt


    def generate_transition_prompt(self, current_scene, action):
        prompt = (
            f"In the scene: {current_scene}. "
            f"Player chooses to {action}. Determine if this action triggers a transition to a new scene."
        )
        return prompt

    def generate_dynamic_scene(self, environment, mood, npc, event):
        # Build prompt with base theme and scene-specific details
        prompt = (
            f"{self.base_theme}. Environment: {environment}, mood: {mood}. "
            f"{npc} is present. An event occurs where {event}. "
            "Describe the scene vividly, adding sounds, lighting, and any notable objects."
        )
        
        # Generate and return dynamic scene description
        dynamic_scene = self.generate_response(prompt, max_length=150)
        return dynamic_scene