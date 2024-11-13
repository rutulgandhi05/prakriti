import torch
import logging

from outlines import models
from transformers import AutoModelForCausalLM, AutoTokenizer

from engine import Erin


logging.basicConfig(
    filename="game_logs.txt",
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def main(model):
    
    
    erin = Erin(model)

    res = erin.prompt("Hi How are you Erin?")

    print(res)

if __name__ == "__main__":

    model_name = "Gigax/NPC-LLM-7B"
    llm = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
 
    model = models.Transformers(llm, tokenizer)
    main(model)

    del model
    del tokenizer
    torch.cuda.empty_cache()
