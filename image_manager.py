# image_manager.py
import os
import torch

from PIL import Image
from diffusers import StableDiffusionXLPipeline


class ImageManager:
    def __init__(self, save_directory="generated_images"):
        """
        Initialize the ImageManager for local SDXL model generation.
        
        Args:
            save_directory (str): Directory to save generated images.
        """
        self.save_directory = save_directory
        os.makedirs(self.save_directory, exist_ok=True)
        
        # Load the SDXL model
        self.pipeline = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", 
            torch_dtype=torch.float16
        ).to("cuda")

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
        with torch.no_grad():
            image = self.pipeline(prompt=prompt).images[0]

        # Save the image with a unique name based on the scene ID
        image_path = os.path.join(self.save_directory, f"{scene_id}_scene.png")
        image.save(image_path)
        torch.cuda.empty_cache()
        return image_path
