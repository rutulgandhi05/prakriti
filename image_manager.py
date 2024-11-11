# image_manager.py
import os
import torch
import logging

from PIL import Image
from diffusers import StableDiffusionXLPipeline, DiffusionPipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ImageManager:
    def __init__(self, save_directory="generated_images"):
        logging.info("Initializing ImageManager")
        self.save_directory = save_directory
        os.makedirs(self.save_directory, exist_ok=True)

        self.image_index = 1


    def generate_image(self, prompt, scene_id):
        logging.info(f"Generating image for scene ID: {scene_id} with prompt: {prompt}")
        # Generate the image using the prompt
        pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16)
        pipe.to("cuda")
        
    
        image = pipe(prompt=prompt, num_inference_steps=35, guidance_scale=7).images[0]

        # Save the image with a unique name based on the scene ID
        image_path = os.path.join(self.save_directory, f"scene_{scene_id}_{self.image_index}_.png")
        image.save(image_path)
        self.image_index = self.image_index + 1
        
        del pipe
        torch.cuda.empty_cache()

        logging.info(f"Image saved at {image_path}")
        return image_path
