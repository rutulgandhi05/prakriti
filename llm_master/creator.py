import torch
import yaml
import argparse

from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM


torch.cuda.empty_cache()

def config_2_args(path):
    with open(path, 'r') as file:
        yaml_data = yaml.safe_load(file)
    parser = argparse.ArgumentParser(description="Generate args from config")
    for key, value in yaml_data.items():
        parser.add_argument(f'--{key}', type=type(value), default=value)
    
    args = parser.parse_args([])
        
    return args

if __name__ == "__main__":
    args = config_2_args("llm_master/config.yaml")