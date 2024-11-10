# image_manager.py
import os
import torch

from PIL import Image
from diffusers import StableDiffusionXLPipeline, DiffusionPipeline


class ImageManager:
    def __init__(self, save_directory="generated_images"):
        """
        Initialize the ImageManager for local SDXL model generation.
        
        Args:
            save_directory (str): Directory to save generated images.
        """
        self.save_directory = save_directory
        os.makedirs(self.save_directory, exist_ok=True)

        self.image_index = 1


    def generate_image(self, prompt, scene_id):
        """
        Generate an image based on the prompt using SDXL.
        
        Args:
            prompt (str): Text prompt to generate the image.
            scene_id (int): Unique ID for the scene, used in the filename.
        
        Returns:
            str: Path to the saved image file.
        """
        # Generate the image using the prompt
        pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16)
        pipe.to("cuda")
        
    
        image = pipe(prompt=prompt, num_inference_steps=35, guidance_scale=7).images[0]

        # Save the image with a unique name based on the scene ID
        image_path = os.path.join(self.save_directory, f"scene_{scene_id}_{self.image_index}_.png")
        image.save(image_path)

        
        del pipe
        torch.cuda.empty_cache()
        return image_path
