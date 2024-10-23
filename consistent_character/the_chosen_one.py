#!/usr/bin/env python
# coding=utf-8

"""Fine-tuning script for Stable Diffusion XL for text2image with support for LoRA."""

import argparse
import os
import random

import torch
import torch.nn.functional as F
import torch.utils.checkpoint

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from datasets import load_dataset
from torchvision import transforms
from transformers import AutoTokenizer, PretrainedConfig, CLIPTextModel

from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionXLPipeline, UNet2DConditionModel
from diffusers.loaders import text_encoder_lora_state_dict
from diffusers.models.lora import LoRALinearLayer
from diffusers.optimization import get_scheduler

from PIL import Image



logger = get_logger(__name__)


# Define the templates for generating textual inversion prompts
imagenet_templates_small = [
    "a photo of a {}",
    "a rendering of a {}",
    "a cropped photo of the {}",
    "the photo of a {}",
    "a photo of a clean {}",
    "a photo of a dirty {}",
    "a dark photo of the {}",
    "a photo of my {}",
    "a photo of the cool {}",
    "a close-up photo of a {}",
    "a bright photo of the {}",
    "a cropped photo of a {}",
    "a photo of the {}",
    "a good photo of the {}",
    "a photo of one {}",
    "a close-up photo of the {}",
    "a rendition of the {}",
    "a photo of the clean {}",
    "a rendition of a {}",
    "a photo of a nice {}",
    "a good photo of a {}",
    "a photo of the nice {}",
    "a photo of the small {}",
    "a photo of the weird {}",
    "a photo of the large {}",
    "a photo of a cool {}",
    "a photo of a small {}",
]

imagenet_style_templates_small = [
    "a painting in the style of {}",
    "a rendering in the style of {}",
    "a cropped painting in the style of {}",
    "the painting in the style of {}",
    "a clean painting in the style of {}",
    "a dirty painting in the style of {}",
    "a dark painting in the style of {}",
    "a picture in the style of {}",
    "a cool painting in the style of {}",
    "a close-up painting in the style of {}",
    "a bright painting in the style of {}",
    "a cropped painting in the style of {}",
    "a good painting in the style of {}",
    "a close-up painting in the style of {}",
    "a rendition in the style of {}",
    "a nice painting in the style of {}",
    "a small painting in the style of {}",
    "a weird painting in the style of {}",
    "a large painting in the style of {}",
]

def save_progress(text_encoder, placeholder_token_ids, accelerator, args, save_path, safe_serialization=True):
    """Save the learned embeddings during training."""
    print("Saving embeddings")
    learned_embeds = (
        accelerator.unwrap_model(text_encoder)
        .get_input_embeddings()
        .weight[min(placeholder_token_ids): max(placeholder_token_ids) + 1]
    )
    learned_embeds_dict = {args.placeholder_token: learned_embeds.detach().cpu()}

    if safe_serialization:
        torch.save(learned_embeds_dict, save_path)


def config_2_args(path):
    """Load and parse the config YAML into args for training."""
    import yaml
    with open(path, 'r') as file:
        yaml_data = yaml.safe_load(file)
    parser = argparse.ArgumentParser(description="Generate args from config")
    for key, value in yaml_data.items():
        parser.add_argument(f'--{key}', type=type(value), default=value)
    
    args = parser.parse_args([])
    return args


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"):
    """Dynamically import the correct text encoder model class from the model name."""
    text_encoder_config = PretrainedConfig.from_pretrained(pretrained_model_name_or_path, subfolder=subfolder, revision=revision)
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        return CLIPTextModel
    else:
        raise ValueError(f"{model_class} is not supported.")


def adjust_learning_rate(optimizer, character_consistency, threshold, high_lr, low_lr):
    """Dynamically adjust the learning rate based on character consistency."""
    if character_consistency < threshold:
        for param_group in optimizer.param_groups:
            param_group['lr'] = high_lr
        print(f"Character consistency below threshold. Using higher learning rate: {high_lr}")
    else:
        for param_group in optimizer.param_groups:
            param_group['lr'] = low_lr
        print(f"Character consistency above threshold. Using lower learning rate: {low_lr}")


def unet_attn_processors_state_dict(unet):
    """
    Returns a state dict containing just the attention processor parameters from the UNet.
    """
    attn_processors = unet.attn_processors

    attn_processors_state_dict = {}

    for attn_processor_key, attn_processor in attn_processors.items():
        for parameter_key, parameter in attn_processor.state_dict().items():
            attn_processors_state_dict[f"{attn_processor_key}.{parameter_key}"] = parameter

    return attn_processors_state_dict


def fine_tune_model(args, loop):
    """
    Fine-tune the model on the provided training data.
    """
    args.output_dir = args.output_dir_per_loop
    args.logging_dir = "logs"
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    args.train_data_dir = args.train_data_dir_per_loop

    print(f"Starting fine-tuning for loop {loop}...")

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    # Set up tokenizers
    tokenizer_one = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer", use_fast=False)
    tokenizer_two = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer_2",  use_fast=False)

    # Load the text encoders
    text_encoder_cls_one = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path)
    text_encoder_cls_two = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, subfolder="text_encoder_2")

    text_encoder_one = text_encoder_cls_one.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    text_encoder_two = text_encoder_cls_two.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder_2")

    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae" )
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")

    # Placeholder tokens and embeddings setup
    placeholder_tokens = [args.placeholder_token]
    additional_tokens = [f"{args.placeholder_token}_{i}" for i in range(1, args.num_vectors)]
    placeholder_tokens += additional_tokens

    num_added_tokens_one = tokenizer_one.add_tokens(placeholder_tokens)
    num_added_tokens_two = tokenizer_two.add_tokens(placeholder_tokens)

    token_ids_one = tokenizer_one.convert_tokens_to_ids(placeholder_tokens)
    token_ids_two = tokenizer_two.convert_tokens_to_ids(placeholder_tokens)

    # Check if initializer_token is a single token or a sequence of tokens
    if len(token_ids_one) > 1:
        raise ValueError("The initializer token must be a single token.")


    # Resize token embeddings
    text_encoder_one.resize_token_embeddings(len(tokenizer_one))
    text_encoder_two.resize_token_embeddings(len(tokenizer_two))

    token_embeds_one = text_encoder_one.get_input_embeddings().weight.data
    token_embeds_two = text_encoder_two.get_input_embeddings().weight.data

    # Initialize embeddings with the initializer token
    with torch.no_grad():
        for token_id in token_ids_one:
            token_embeds_one[token_id] = token_embeds_one[token_ids_one[0]].clone()
        for token_id in token_ids_two:
            token_embeds_two[token_id] = token_embeds_two[token_ids_two[0]].clone()


    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=torch.float32)
    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)



    # Insert LoRA layers into the attention modules of UNet
    unet_lora_parameters = []
    for attn_processor_name, attn_processor in unet.attn_processors.items():
        attn_module = unet
        # Traverse through the module names to reach the target layer (e.g., to_q, to_k, to_v)
        for n in attn_processor_name.split(".")[:-1]:
            attn_module = getattr(attn_module, n)

        # Set LoRA layers for each attention module (to_q, to_k, to_v, to_out)
        attn_module.to_q.set_lora_layer(LoRALinearLayer(attn_module.to_q.in_features, attn_module.to_q.out_features, rank=args.rank))
        attn_module.to_k.set_lora_layer(LoRALinearLayer(attn_module.to_k.in_features, attn_module.to_k.out_features, rank=args.rank))
        attn_module.to_v.set_lora_layer(LoRALinearLayer(attn_module.to_v.in_features, attn_module.to_v.out_features, rank=args.rank))
        attn_module.to_out[0].set_lora_layer(LoRALinearLayer(attn_module.to_out[0].in_features, attn_module.to_out[0].out_features, rank=args.rank))

        unet_lora_parameters.extend(attn_module.to_q.lora_layer.parameters())
        unet_lora_parameters.extend(attn_module.to_k.lora_layer.parameters())
        unet_lora_parameters.extend(attn_module.to_v.lora_layer.parameters())
        unet_lora_parameters.extend(attn_module.to_out[0].lora_layer.parameters())

    # Freeze non-LoRA parameters
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    unet.requires_grad_(False)

    # LoRA parameters remain trainable
    for param in unet_lora_parameters:
        param.requires_grad = True

    # Optimizer setup for LoRA parameters only
    optimizer = torch.optim.AdamW(
        unet_lora_parameters, 
        lr=args.learning_rate, 
        betas=(args.adam_beta1, args.adam_beta2), 
        weight_decay=args.adam_weight_decay, 
        eps=args.adam_epsilon
    )

    
    # Optimizer setup with dynamic learning rate adjustment
    high_lr = args.high_learning_rate
    low_lr = args.low_learning_rate
    threshold = args.consistency_threshold


    # Training data setup
    train_dataset = TextualInversionDataset(
        args=args,
        data_root=args.train_data_dir,
        tokenizer_one=tokenizer_one,
        tokenizer_two=tokenizer_two,
        size=args.resolution,
        placeholder_token=(" ".join(tokenizer_one.convert_ids_to_tokens(token_ids_one))),
        repeats=args.repeats,
        learnable_property=args.learnable_property,
        center_crop=args.center_crop,
        set="train",
    )

    # DataLoader setup
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers
    )

    # Scheduler setup
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Prepare for training
    unet, text_encoder_one, text_encoder_two, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, text_encoder_one, text_encoder_two, optimizer, train_dataloader, lr_scheduler
    )

    # Start training with dynamic learning rate adjustment
    progress_bar = range(args.max_train_steps)
    for step in progress_bar:
        adjust_learning_rate(optimizer, args.character_consistency, threshold, high_lr, low_lr)  # Adjust LR

        for batch in train_dataloader:
            with accelerator.accumulate(unet):
                pixel_values = batch["pixel_values"].to(dtype=weight_dtype)

                model_input = vae.encode(pixel_values).latent_dist.sample() * vae.config.scaling_factor

                noise = torch.randn_like(model_input)
                timesteps = torch.randint(
                    0, DDPMScheduler.config.num_train_timesteps, (model_input.shape[0],), device=model_input.device
                ).long()

                noisy_model_input = DDPMScheduler.add_noise(model_input, noise, timesteps)

                # Use added conditions
                unet_added_conditions = {
                    "text_embeds": batch["text_embeds"]
                }
                model_pred = unet(
                    noisy_model_input, timesteps, prompt_embeds=batch["text_embeds"], added_cond_kwargs=unet_added_conditions
                ).sample

                target = noise if DDPMScheduler.config.prediction_type == "epsilon" else DDPMScheduler.get_velocity(model_input, noise, timesteps)
                loss = F.mse_loss(model_pred.float(), target.float())

                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

    # Save the model after training
    print(f"Training for loop finished, saving model...")
    save_path = os.path.join(args.output_dir, f"checkpoint-loop-{args.loop}")
    accelerator.save_state(save_path)

     # Save LoRA layers after training
    StableDiffusionXLPipeline.save_lora_weights(
        save_directory=args.output_dir,
        unet_lora_layers=unet_attn_processors_state_dict(unet),
        text_encoder_lora_layers=text_encoder_lora_state_dict(text_encoder_one),
        text_encoder_2_lora_layers=text_encoder_lora_state_dict(text_encoder_two),
    )

    print(f"Model saved at {save_path}.")


class TextualInversionDataset(torch.utils.data.Dataset):
    """
    Dataset for textual inversion fine-tuning.
    """
    def __init__(self, args, data_root, tokenizer_one, tokenizer_two, size, placeholder_token, repeats=100, learnable_property="object", center_crop=False, set="train"):
        self.args = args
        self.data_root = data_root
        self.tokenizer_one = tokenizer_one
        self.tokenizer_two = tokenizer_two
        self.placeholder_token = placeholder_token
        self.center_crop = center_crop
        self.size = size
        self.repeats = repeats

        self.image_paths = [os.path.join(data_root, file_path) for file_path in os.listdir(data_root)]
        self.num_images = len(self.image_paths)

        self.templates = imagenet_templates_small if learnable_property == "object" else imagenet_style_templates_small
        self.flip_transform = transforms.RandomHorizontalFlip(p=0.5)

    def __len__(self):
        return self.num_images * self.repeats

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index % self.num_images])

        if self.center_crop:
            image = transforms.CenterCrop(self.size)(image)
        else:
            image = transforms.RandomResizedCrop(self.size)(image)
        image = transforms.ToTensor()(image)

        text = random.choice(self.templates).format(self.placeholder_token)
        input_ids_one = self.tokenizer_one(text, return_tensors="pt", padding="max_length", truncation=True).input_ids[0]
        input_ids_two = self.tokenizer_two(text, return_tensors="pt", padding="max_length", truncation=True).input_ids[0]

        return {
            "pixel_values": image,
            "text_embeds": (input_ids_one, input_ids_two),
        }


if __name__ == "__main__":
    args = config_2_args("thechosenone/config/captain.yaml")
    args.revision=None
    fine_tune_model(args)
