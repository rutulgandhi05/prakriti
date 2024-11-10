# llm_engine.py
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from huggingface_hub import login

login(token="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")

class LLMEngine:
    def __init__(self, model_name="mistralai/Mistral-7B-v0.1"):
        # Load the Mistral 7B model and tokenizer
        self.model_name = model_name

    def generate_response(self, player_input):
        """
        Generate an NPC response based on the player input and scene context.
        
        Args:
            player_input (str): The player's input.
            context (str): The current scene context.

        Returns:
            str: Generated response from the LLM.
        """
        model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Construct the prompt for the model
        prompt = player_input

        # Tokenize and generate the response
        inputs = tokenizer(prompt, return_tensors="pt").to('cuda')
        outputs = model.generate(inputs.input_ids, max_length=150, temperature=0.7)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        del model
        del tokenizer
        torch.cuda.empty_cache()
        return response.strip()

