import os 
import torch
import random
import numpy as np
import torchvision.transforms as T

from tqdm import auto
from PIL import Image,ImageEnhance
from diffusers import DiffusionPipeline,DDIMScheduler,DDPMScheduler


def save_XLembedding(emb, embedding_file, path):
        torch.save(emb, path+embedding_file)

    
def set_XLembedding(base, emb, token):
    with torch.no_grad():            
        # Embeddings[tokenNo] to learn
        tokens=base.components["tokenizer"].encode(token)
        assert len(tokens)==3, "token is not a single token in 'tokenizer'"
        tokenNo=tokens[1]
        tokens=base.components["tokenizer_2"].encode(token)
        assert len(tokens)==3, "token is not a single token in 'tokenizer_2'"
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



def main(args):
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

    p1="The {} doll at the beach"
    p2="The 3D rendering of a group of {} figurines dressed in red-striped bathing suits having fun at the beach"
    p3="The 3D rendering of a group of {} figurines dressed in dirndl wearing sunglasses drinking beer and having fun at the oktoberfest"
    negative_prompt="disfigure kitsch ugly oversaturated deformed mutation blurry mutated duplicate malformed cropped, bad anatomy, outof focus frame, poorly drawn face, low quality, cloned face, deformed face, squint eyes, malformed hand, fused fingers, crooked arm leg, missing disconnect arm leg"
    n_steps=40
    high_noise_frac=.75

    for seed,sample_prompt in zip([20,30,40,1,8,9,45,75,90],[p1,p1,p1,p2,p2,p2,p3,p3,p3]): 
        prompt=sample_prompt.format(learned)
        with torch.no_grad():    
            torch.manual_seed(seed)
            image = base(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=n_steps,
                denoising_end=high_noise_frac,
                output_type="latent"
            ).images

            image = refiner(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=n_steps,
                denoising_start=high_noise_frac,
                image=image,
            ).images[0]

            output_folder = f"data/inference_results/{args.character_name}/txt_inv"
            os.makedirs(output_folder, exist_ok=True)
            image.save( os.path.join(output_folder, "{}.png".format(seed)))
           