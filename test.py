from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from huggingface_hub import login

torch.cuda.empty_cache()

login(token="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")

# Load the tokenizer and model
model_path = "nvidia/Mistral-NeMo-Minitron-8B-Base"
tokenizer = AutoTokenizer.from_pretrained(model_path)

device = 'cuda'
dtype = torch.bfloat16
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype)

# Prepare the input text
prompt = 'Hey how are you? Can i go from london to paris by train?'
inputs = tokenizer.encode(prompt, return_tensors='pt').to(model.device)

# Generate the output
outputs = model.generate(inputs, max_length=20)

# Decode and print the output
output_text = tokenizer.decode(outputs[0])
print(output_text)