from diffusers import DiffusionPipeline
import torch
import argparse
import yaml
import os

def config_2_args(path):
    """
    Parse configuration YAML file to command-line arguments.
    :param path: Path to the YAML config file.
    :return: Parsed arguments as an argparse.Namespace.
    """
    with open(path, 'r') as file:
        yaml_data = yaml.safe_load(file)
    parser = argparse.ArgumentParser(description="Generate args from yaml")
    for key, value in yaml_data.items():
        parser.add_argument(f'--{key}', type=type(value), default=value)
    
    args = parser.parse_args([])
    return args

def generate_game_image(prompt, config_path="consistent_character/config/erin.yaml", loop=0, character_name="default_character"):
    """
    Generate an image using Stable Diffusion for a specific character and quest.
    :param prompt: Prompt text for the image generation.
    :param config_path: Path to the YAML configuration file.
    :param loop: Loop count or checkpoint number.
    :param character_name: Name of the character for folder organization.
    :return: The generated image.
    """
    # Load configuration from YAML file
    args = config_2_args(config_path)
    
    # Set model path and load pipeline with LoRA weights
    model_path = args.get("model_path", "default_model_path")
    pipe = DiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
    pipe.to("cuda")
    
    # Load LoRA weights dynamically
    checkpoint_path = os.path.join(model_path, f"checkpoint-{args.get('checkpointing_steps', 500) * args.get('num_train_epochs', 3)}")
    if os.path.exists(checkpoint_path):
        pipe.load_lora_weights(checkpoint_path)
    
    # Set up output directory for the images
    output_folder = args.get("output_folder", f"data/inference_results/{character_name}")
    os.makedirs(output_folder, exist_ok=True)
    
    # Generate the image
    image = pipe(prompt, num_inference_steps=35, guidance_scale=7.5).images[0]
    output_path = os.path.join(output_folder, f"{character_name}_loop_{loop}.png")
    image.save(output_path)
    return image

# Keeping loop_inference unchanged per request
def loop_inference(loop, prompt_postfix):
    """
    Generate a series of images in a loop for predefined prompt postfixes.
    :param loop: Loop count used for file naming.
    :param prompt_postfix: Text added to the prompt.
    :return: Saves generated images in output folders.
    """
    args = config_2_args("consistent_character/config/erin.yaml")

    model_path = os.path.join(args.output_dir, args.character_name, str(loop))
    pipe = DiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
    pipe.to("cuda")
    pipe.load_lora_weights(os.path.join(model_path, f"checkpoint-{args.checkpointing_steps * args.num_train_epochs}"))

    prompt_postfix = prompt_postfix
    image_postfix = prompt_postfix.replace(" ", "_")

    # Create output folder
    output_folder = f"data/inference_results/{args.character_name}/{loop}"
    os.makedirs(output_folder, exist_ok=True)

    # Generate prompt and image
    prompt = f"A  photo of ({args.placeholder_token}::4) {prompt_postfix}."
    image = pipe(prompt, num_inference_steps=35, guidance_scale=8).images[0]
    image.save(os.path.join(output_folder, f"{args.character_name}_{image_postfix}_loop_{loop}.png"))

if __name__ == "__main__":
    prompt_postfixs = ["in a forest", "standing infront of castle", "holding a red flag", "sitting on a bench", "holding a sword, background has dead warriors","in a simple white background"]

    for prompt_postfix in prompt_postfixs:
            loop_inference(0, prompt_postfix)
        
