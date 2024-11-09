import os 
import yaml
import numpy as np
import torch
import argparse
import random
import torchvision.transforms as T

from tqdm import auto
from PIL import Image,ImageEnhance
from diffusers import DiffusionPipeline,DDIMScheduler,DDPMScheduler
from generation_prompts import prompts


def config_2_args(path):
    with open(path, 'r') as file:
        yaml_data = yaml.safe_load(file)
    parser = argparse.ArgumentParser(description="Generate args from config")
    for key, value in yaml_data.items():
        parser.add_argument(f'--{key}', type=type(value), default=value)
    
    args = parser.parse_args([])
        
    return args


def save_XLembedding(emb, embedding_file, path):
        torch.save(emb, path+embedding_file)

    
def set_XLembedding(base, emb, token):
    with torch.no_grad():            
        # Embeddings[tokenNo] to learn
        tokens=base.components["tokenizer"].encode(token)

        tokenNo=tokens[1]
        tokens=base.components["tokenizer_2"].encode(token)

        tokenNo2=tokens[1]
        embs=base.components["text_encoder"].text_model.embeddings.token_embedding.weight
        embs2=base.components["text_encoder_2"].text_model.embeddings.token_embedding.weight
        assert embs[tokenNo].shape==emb["emb"].shape, "different 'text_encoder'"
        assert embs2[tokenNo2].shape==emb["emb2"].shape, "different 'text_encoder_2'"
        embs[tokenNo]=emb["emb"].to(embs.dtype).to(embs.device)
        embs2[tokenNo2]=emb["emb2"].to(embs2.dtype).to(embs2.device)


def load_XLembedding(base, token, embedding_file, path):
    emb=torch.load(path+embedding_file)
    set_XLembedding(base, emb, token)



def inference(args, prompt_postfix):
    output_folder = f"data/inference_results/3r1n_mod/txt_inv"
    os.makedirs(output_folder, exist_ok=True)

    base = DiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path, 
        torch_dtype=torch.float16, #torch.bfloat16
        variant="fp32", 
        use_safetensors=True,
        add_watermarker=False,
        )
    
    base.enable_xformers_memory_efficient_attention()
    torch.set_grad_enabled(False)
    
    _=base.to("cuda")

    refiner = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        text_encoder_2=base.text_encoder_2,  
        vae=base.vae,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
        add_watermarker=False,
    )
    refiner.enable_xformers_memory_efficient_attention()
    _=refiner.to("cuda")


    learned=args.placeholder_token
    embs_path=args.teacher_output_dir
    emb_file=f"{args.character_name}.pt"

    load_XLembedding(base,token=learned,embedding_file=emb_file,path=embs_path)

    prompt_postfix = prompt_postfix
    prompt = f"A photo of {args.placeholder_token} {prompt_postfix}."
    n_prompt = "(glasses:1.2), young, teen, child, (deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, (mutated hands and fingers:1.4), disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation, tattoo"

    n_steps=40
    high_noise_frac=.75
    
    with torch.no_grad():    
        torch.manual_seed(args.seed)
        image = base(
            prompt=prompt,
            negative_prompt=n_prompt,
            num_inference_steps=n_steps,
            denoising_end=high_noise_frac,
            output_type="latent",
            guidance_scale =7.5
        ).images

        image = refiner(
            prompt=prompt,
            negative_prompt=n_prompt,
            num_inference_steps=n_steps,
            denoising_start=high_noise_frac,
            image=image,
        ).images[0]

        
        image.save( os.path.join(output_folder, "{}.png".format(prompt.replace(" ", "_"))))


def training_images(args, prompt):
    output_folder = f"{args.backup_data_dir_root}/{args.character_name}/0"
    os.makedirs(output_folder, exist_ok=True)

    base = DiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path, 
        torch_dtype=torch.float16, #torch.bfloat16
        variant="fp32", 
        use_safetensors=True,
        add_watermarker=False,
        )
    
    base.enable_xformers_memory_efficient_attention()
    torch.set_grad_enabled(False)
    
    _=base.to("cuda")

    refiner = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        text_encoder_2=base.text_encoder_2,  
        vae=base.vae,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
        add_watermarker=False,
    )
    refiner.enable_xformers_memory_efficient_attention()
    _=refiner.to("cuda")


    learned=args.placeholder_token
    embs_path=args.teacher_output_dir
    emb_file=f"{args.character_name}.pt"

    prompt = prompt.format(learned)

    load_XLembedding(base,token=learned,embedding_file=emb_file,path=embs_path)

    n_prompt = "(glasses:1.2), young, teen, child, (deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, (mutated hands and fingers:1.4), disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation, tattoo"

    n_steps=40
    high_noise_frac=.75
    
    print(prompt)

    with torch.no_grad():    
        torch.manual_seed(args.seed)
        image = base(
            prompt=prompt,
            negative_prompt=n_prompt,
            num_inference_steps=n_steps,
            denoising_end=high_noise_frac,
            output_type="latent"
        ).images

        image = refiner(
            prompt=prompt,
            negative_prompt=n_prompt,
            num_inference_steps=n_steps,
            denoising_start=high_noise_frac,
            image=image,
        ).images[0]

        
        image.save( os.path.join(output_folder, "{}.png".format(prompt.replace(" ", "_"))))
        torch.cuda.empty_cache()



if __name__ == "__main__":
    args = config_2_args("consistent_character/config/erin.yaml")

    """ prompt_postfixs = ["with dense green forest in background", "ERIN written with white color in black backround", "sitting on a bench with simple bricks wall in background", "eating food, with grey wall in background"]

    for prompt_postfix in prompt_postfixs:
            
        inference(args=args, prompt_postfix=prompt_postfix) """
    
    for prompt in prompts[:50]:
        training_images(args=args, prompt=prompt)