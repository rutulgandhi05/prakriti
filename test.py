from diffusers import DiffusionPipeline
import torch
import argparse
import yaml
import os



loop = 1
#model_path = os.path.join("output", "agent_fox", (loop))
pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16)
pipe.to("cuda")
pipe.load_lora_weights("latent-consistency/lcm-lora-sdxl")

prompt_postfix = " sitting on a rocket"
image_postfix = prompt_postfix.replace(" ", "_")

# create folder
output_folder = f"generated_images/agent_fox"
os.makedirs(output_folder, exist_ok= True)

# remember to use the place holader here
prompt = f"A photo of <$V$> {prompt_postfix}."
image = pipe(prompt, num_inference_steps=35, guidance_scale=7.5).images[0]
image.save("generated_images/agent_fox/{image_postfix}.png")