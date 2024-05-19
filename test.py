import os
import torch
from PIL import Image
from openai import OpenAI
from transformers import pipeline
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, StableDiffusionPipeline


client = OpenAI(
    # This is the default and can be omitted
    api_key='sk-proj-qDQrC201Ia6GTICngFmbT3BlbkFJPWGyoLDdUOYMKCgIBS8D',
)

controlnet_model = os.path.join("models", "control_v11p_sd15_normalbae.pth")

# Enhance the description using GPT model
def enhance_description(prompt):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=[
            {"role": "system", "content": "You re-write the text with details in a way that it can be passed to a model to create an Image."},
            {"role": "user", "content": prompt}
        ],
    )
    return response.choices[0].message.content


def generate_base_image(description):
    # Load the Stable Diffusion model
    model_id = "CompVis/stable-diffusion-v1-4"
    pipe = StableDiffusionPipeline.from_pretrained(model_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipe.to(device)
    
    # Generate the base image
    image = pipe(description).images[0]
    return image


# Generate an image using ControlNet
def generate_image(description, base_image):
    # Load ControlNet and Stable Diffusion models
    control_model = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_normalbae")
    pipe = StableDiffusionControlNetPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", controlnet=control_model)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipe.to(device)
    
    # Generate the image
    image = pipe(prompt=description, image=base_image).images[0]
    return image

if __name__ == "__main__":
    # User input prompt
    user_prompt = "A person drinking coffee in his living room. The person is sitting on a gaming chair."
    
    # Enhance the description using GPT model
    detailed_description = enhance_description(user_prompt)
    print("Enhanced Description:", detailed_description)

    # Generate the base image
    base_image = generate_base_image(detailed_description)
    base_image.save("base_futuristic_city.png")
    
    # Generate and save the image
    generated_image = generate_image(detailed_description, base_image)
    generated_image.save("enhanced_futuristic_city.png")
    generated_image.show()
