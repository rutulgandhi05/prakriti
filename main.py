import openai
import torch
import os
import asyncio
from openai import OpenAI
from diffusers import StableDiffusionPipeline
from PIL import Image


client = OpenAI(
    # This is the default and can be omitted
    api_key='sk-proj-qDQrC201Ia6GTICngFmbT3BlbkFJPWGyoLDdUOYMKCgIBS8D',
)

def generate_detailed_description(prompt):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
    )
    return response.choices[0].message

# User input prompt
user_prompt = "A room wwith a person sitting on a sofa."
detailed_description = generate_detailed_description(user_prompt)
print("Detailed Description:", detailed_description)


'''
# Load Stable Diffusion model
model_id = "CompVis/stable-diffusion-v1-4"
pipeline = StableDiffusionPipeline.from_pretrained(model_id)
device = "cuda" if torch.cuda.is_available() else "cpu"
pipeline = pipeline.to(device)

def generate_image(description):
    image = pipeline(description).images[0]
    return image

# Generate and save image
generated_image = generate_image(detailed_description)
generated_image.save("futuristic_city.png")
generated_image.show()
'''