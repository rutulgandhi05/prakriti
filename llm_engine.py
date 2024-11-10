# llm_engine.py
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class LLMEngine:
    def __init__(self, model_name="mistralai/Mistral-7B-v0.1"):
        # Load the Mistral 7B model and tokenizer
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def generate_response(self, player_input, context):
        """
        Generate an NPC response based on the player input and scene context.
        
        Args:
            player_input (str): The player's input.
            context (str): The current scene context.

        Returns:
            str: Generated response from the LLM.
        """
        # Construct the prompt for the model
        prompt = f"""
        You are an NPC in a narrative-driven game. The current scene context is as follows: {context}
        
        The player says: "{player_input}"
        
        Respond appropriately as the NPC.
        """
        
        # Tokenize and generate the response
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(inputs.input_ids, max_length=150, temperature=0.7)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return response.strip()

# Example test for LLMEngine
if __name__ == "__main__":
    llm_engine = LLMEngine()
    
    # Define sample context and player input
    context = "The player stands before Eldara, an ancient forest guardian, in a mystical forest at dusk."
    player_input = "I ask Eldara about the powers of the artifact."
    
    # Generate and print the response
    response = llm_engine.generate_response(player_input, context)
    print("NPC Response:", response)