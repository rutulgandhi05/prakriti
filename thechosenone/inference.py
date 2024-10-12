from diffusers import DiffusionPipeline
import torch
import argparse
import yaml
import os

def config_2_args(path):
    with open(path, 'r') as file:
        yaml_data = yaml.safe_load(file)
    parser = argparse.ArgumentParser(description="Generate args from yaml")
    for key, value in yaml_data.items():
        parser.add_argument(f'--{key}', type=type(value), default=value)
    
    args = parser.parse_args([])
    return args

def generate_game_image(prompt, config_path="thechosenone/config/captain.yaml", loop=0):
    """
    Generate an image using Stable Diffusion for a specific character and quest.
    :param character_name: The name of the character or NPC for the prompt.
    :param prompt_postfix: The dynamic part of the prompt based on the quest or scene.
    :param config_path: Path to the YAML configuration file.
    :param loop: Loop count or checkpoint number.
    :return: None (saves the generated image to a specific folder).
    """
    # Load configuration from YAML file
    args = config_2_args(config_path)
    
    # Set model path and load pipeline with LoRA weights
    model_path = args["model_path"]
    pipe = DiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
    pipe.to("cuda")
    
    # Load LoRA weights for the character
    checkpoint_path = os.path.join(model_path, f"checkpoint-{args['checkpointing_steps'] * args['num_train_epochs']}")
    pipe.load_lora_weights(checkpoint_path)
    
    # Set up output directory for the images
    output_folder = args["output_folder"]
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Generate the image
    image = pipe(prompt, num_inference_steps=35, guidance_scale=7.5).images[0]
    return image


def loop_inference(loop):
    args = config_2_args("thechosenone/config/captain.yaml")

    model_path = os.path.join(args.output_dir, args.character_name, str(loop))
    pipe = DiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
    pipe.to("cuda")
    pipe.load_lora_weights(os.path.join(model_path, f"checkpoint-{args.checkpointing_steps * args.num_train_epochs}"))

    prompt_postfix = "sitting on a rocket"
    image_postfix = prompt_postfix.replace(" ", "_")

    # create folder
    output_folder = f"data/inference_results/{args.character_name}"
    os.makedirs(output_folder, exist_ok=True)

    # remember to use the place holader here
    prompt = f"A photo of {args.placeholder_token}{prompt_postfix}."
    image = pipe(prompt, num_inference_steps=35, guidance_scale=7.5).images[0]
    image.save(os.path.join(output_folder, f"{args.character_name}_{image_postfix}_loop_{loop}.png"))


if __name__ == "__main__":
    for i in range(5):
        loop_inference(i)

        