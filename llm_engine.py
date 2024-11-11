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

    def generate_response(self, player_input):
        logging.info(f"Generating response with prompt: {prompt}")
        model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.float16).to('cuda')
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Construct the prompt for the model
        prompt = player_input

        # Tokenize and generate the response
        inputs = tokenizer([prompt], return_tensors="pt").to('cuda')
    
        outputs = model.generate(**inputs, max_length=500, temperature=0.7)
       
        tokenizer.batch_decode(outputs)[0]
        response = tokenizer.batch_decode(outputs)[0].strip()
        logging.info(f"Generated response: {response}")
        del model
        del tokenizer
        torch.cuda.empty_cache()
        return response

