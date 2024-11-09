import os
import shutil
import yaml
import numpy as np
import torch
import argparse
import random
import torch.utils.checkpoint
import torchvision.transforms as T

from diffusers import StableDiffusionXLPipeline, DiffusionPipeline
from PIL import Image
from diffusers import DDIMScheduler, DPMSolverMultistepScheduler
from the_chosen_one import train as train_pipeline
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist


def train_loop(args, loop_num: int, start_from=0):
    """
    train and load the trained diffusion model, save the images and model file.
    """
    output_dir_base = args.output_dir
    train_data_dir_base = args.train_data_dir
    num_train_epochs = args.num_train_epochs
    checkpointing_steps = args.checkpointing_steps
    
    args.kmeans_center = int(args.num_of_generated_img / args.dsize_c)
    
    # initial pair wise distance
    init_dist = 0

    
    # start looping
    for loop in range(start_from, loop_num):
        print()
        print("###########################################################################")
        print("###########################################################################")
        print(f"[{loop}/{loop_num-1}] Start.")
        print("###########################################################################")
        print("###########################################################################")
        print()
        
        # load dinov2 every epoch, since we clean the model after feature etraction
        dinov2 = load_dinov2()
        
        # load diffusion pipeline every epoch for new training image generation, since we clean the model after feature etraction
        if loop == 0:
            # load from default SDXL config.
            refiner, pipe = load_trained_pipeline(args=args)
        else:
            # Note that these configurations are changned during training.
            # Since the the training is epoch based and we use iterations, the diffuser training script automatically calculate a new epoch according to the iteration and dataset size, thus the predefined epoches will be overrided.
            
            # load model from the output dir in PREVIOUS loop
            refiner, pipe = load_trained_pipeline(model_path=args.output_dir_per_loop, 
                                          load_lora=True, 
                                          lora_path=os.path.join(args.output_dir_per_loop, f"checkpoint-{checkpointing_steps * num_train_epochs}"),
                                          args=args)
        
        # update model output dir for CURRENT loop
        args.output_dir_per_loop = os.path.join(output_dir_base, args.character_name, str(loop))
        
        # set up the training data folder used in training, overwrite and recreate
        args.train_data_dir_per_loop = os.path.join(train_data_dir_base, args.character_name, str(loop))
        if os.path.exists(args.train_data_dir_per_loop):
            shutil.rmtree(args.train_data_dir_per_loop)
        os.makedirs(args.train_data_dir_per_loop, exist_ok=True)
        
        # generate new images
        image_embs = []
        images = []
        for n_img in range(args.num_of_generated_img):
            print(f"[Loop [{loop}/{loop_num-1}], generating image {n_img}/{args.num_of_generated_img - 1}")
            
            # set up different seeds for each image
            torch.manual_seed(n_img * np.random.randint(1000))
            tmp_folder = f"{args.backup_data_dir_root}/{args.character_name}/{loop}"
            
            # we can load the initially generated images from local backup folder
            if loop==0 and os.path.exists(os.path.join(tmp_folder, f"{n_img}.png")):
                image = Image.open(os.path.join(tmp_folder, f"{n_img}.png")).convert('RGB')
            else:
                image = generate_images(pipe, tmp_folder=tmp_folder, placeholder_token=args.placeholder_token, refiner=refiner,n_img=n_img, text_inv_prompt=args.text_inv_prompt, prompt=args.inference_prompt, n_prompt=args.negative_prompt, infer_steps=args.infer_steps)
                
            images.append(image)
            image_embs.append(infer_model(dinov2, image).detach().cpu().numpy())
            
            
        
        # clean up the GPU consumption after inference
        del pipe
        del dinov2
        torch.cuda.empty_cache()
        
        # reshaping
        embeddings = np.array(image_embs)
        embeddings = embeddings.reshape(len(image_embs), -1)
        
        # evaluate convergence
        if loop == 0:
            init_dist = np.mean(cdist(embeddings, embeddings, 'euclidean'))
        else:
            pairwise_distances = np.mean(cdist(embeddings, embeddings, 'euclidean'))
            if pairwise_distances < init_dist * args.convergence_scale:
                print()
                print("###########################################################################")
                print("###########################################################################")
                print(f"Converge at {loop}. Target distance: {init_dist}, current pairwise distance: {pairwise_distances}. Final model saved at {os.path.join(output_dir_base, args.character_name, str(loop - 1))}")
                print("###########################################################################")
                print("###########################################################################")
                print()
                return os.path.join(output_dir_base, args.character_name, str(loop - 1)), 
            else:
                print()
                print("###########################################################################")
                print("###########################################################################")
                print(f"Target distance: {init_dist}, current pairwise distance: {pairwise_distances}.")
                print("###########################################################################")
                print("###########################################################################")
                print()
        # clustering
        centers, labels, elements, images = kmeans_clustering(args, embeddings, images = images)
        kmeans_2D_visualize(args, centers, elements, labels, loop)
        
        # evaluate
        center_norms = np.linalg.norm(centers[labels] - elements, axis=-1, keepdims=True) # each data point subtract its coresponding center
        cohesions = np.zeros(len(np.unique(labels)))
        for label_id in range(len(np.unique(labels))):
            cohesions[label_id] = sum(center_norms[labels == label_id]) / sum(labels == label_id)
        
        # find the most cohesive cluster, and save the corresponding sample
        min_cohesion_label = np.argmin(cohesions)
        idx = np.where(labels == min_cohesion_label)[0]
        for sample_id, sample in enumerate(images):
            if sample_id in idx:
                sample.save(os.path.join(args.train_data_dir_per_loop, f"{sample_id}.png"))
        
        
        print()
        print("###########################################################################")
        print("###########################################################################")
        print(f"[{loop}/{loop_num-1}] Finish.")
        print("###########################################################################")
        print("###########################################################################")
        print()
        

def kmeans_clustering(args, data_points, images = None):
    kmeans = KMeans(n_clusters=args.kmeans_center, init='k-means++', random_state=42)
    kmeans.fit(data_points)
    labels = kmeans.labels_

    unique, counts = np.unique(labels, return_counts=True)
    cluster_counts = dict(zip(unique, counts))

    selected_clusters = [cluster for cluster, count in cluster_counts.items() if count > args.dmin_c]
    selected_centers = kmeans.cluster_centers_[selected_clusters]
    selected_labels = []
    
    selected_labels = [label for label in labels if label in selected_clusters]
    selected_labels = make_continuous(selected_labels)
    selected_labels = np.array(selected_labels)
    
    selected_elements = np.array([data_points[i] for i, label in enumerate(labels) if label in selected_clusters])
    if images:
        selected_images = [images[i] for i, label in enumerate(labels) if label in selected_clusters]
    else:
        selected_images = None

    return selected_centers, selected_labels, selected_elements, selected_images


def make_continuous(lst):
    unique_elements = sorted(set(lst))
    mapping = {elem: i for i, elem in enumerate(unique_elements)}
    return [mapping[elem] for elem in lst]


def kmeans_2D_visualize(args, centers, data, labels, loop_num):
    # visualize 2D t-SNE results
    plt.figure(figsize=(20, 16))
    tsne = TSNE(n_components=2, random_state=42, perplexity=len(data) - 1)
    embeddings_2d = tsne.fit_transform(data)
    
    for i in range(args.kmeans_center):
        cluster_points = np.array(embeddings_2d[labels==i])
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {i + 1}", s=100)
    plt.savefig(f"data/{args.character_name}/_KMeans_res_Loop_{loop_num}.png")
    
        
def compare_features(image_features, cluster_centroid):
    # Calculate the Euclidean distance between the two feature vectors
    distance = np.linalg.norm(image_features - cluster_centroid)
    return distance


def prepare_init_images(source_path, target_root_path):
    img_out_base = target_root_path
    init_loop_img_fdr = os.path.join(img_out_base, "0")
    os.makedirs(init_loop_img_fdr, exist_ok=True)
    
    for item in os.listdir(source_path):
        src_path = os.path.join(source_path, item)
        dest_path = os.path.join(init_loop_img_fdr, item)
        
        shutil.copy2(src_path, dest_path)
        print(f"Copied {src_path} to {dest_path}")


def load_trained_pipeline(args, model_path = None, load_lora=True, lora_path=None):
    """
    load the diffusion pipeline according to the trained model
    """
    if model_path is not None:
        ddpm = DPMSolverMultistepScheduler.from_pretrained(model_path, subfolder="scheduler")
        pipe = StableDiffusionXLPipeline.from_pretrained(model_path, torch_dtype=torch.float16, use_safetensors=True)
        
        if load_lora:
            pipe.load_lora_weights(lora_path)

        pipe.scheduler = ddpm

    else:
        ddpm = DPMSolverMultistepScheduler.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="scheduler", use_karras_sigmas=True)
        pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", use_safetensors=True)
        pipe.scheduler = ddpm

        embs_path=args.teacher_output_dir
        emb_file=f"{args.character_name}.pt"
        emb=torch.load(embs_path+emb_file)
        pipe.load_textual_inversion(emb["emb2"], token=args.character_name, text_encoder=pipe.text_encoder_2, tokenizer=pipe.tokenizer_2)
        pipe.load_textual_inversion(emb["emb"], token={args.character_name}, text_encoder=pipe.text_encoder, tokenizer=pipe.tokenizer)

        refiner = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            text_encoder_2=pipe.text_encoder_2,
            vae=pipe.vae,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        ).to("cuda")

    
    pipe.to("cuda")
    return refiner, pipe


def config_2_args(path):
    with open(path, 'r') as file:
        yaml_data = yaml.safe_load(file)
    parser = argparse.ArgumentParser(description="Generate args from config")
    for key, value in yaml_data.items():
        parser.add_argument(f'--{key}', type=type(value), default=value)
    
    args = parser.parse_args([])
        
    return args


def infer_model(model, image):
    transform = T.Compose([
        T.Resize((518, 518)),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    image = transform(image).unsqueeze(0).cuda()
    cls_token = model(image, is_training=False)
    return cls_token


def generate_images(pipe: StableDiffusionXLPipeline, tmp_folder, n_img, placeholder_token, text_inv_prompt, refiner, n_prompt: str, infer_steps, guidance_scale=7.5):
    """
    use the given StableDiffusionXLPipeline, generate N images for the same character
    return: image, in PIL
    """
    x_values = ["an extreme closeup", "a medium closeup", "a closeup", "a medium shot", "a full body"]
    y_values = ["front shot", "rear angle", "side angle", "shot from above", "low angle shot"]
 
    #new_prompt = random.choice(x_values)+" "+ random.choice(y_values)+" "+prompt
    new_prompt = f" {random.choice(x_values)} {random.choice(y_values)} of ({placeholder_token}::1.8), with neutral background"
    print(new_prompt)

    image_filename = f"{random.choice(x_values)} {text_inv_prompt} ({n_img}).png".replace(' ', '_')

    image = pipe(prompt=new_prompt, 
                num_inference_steps=infer_steps,
                guidance_scale=guidance_scale, 
                negative_prompt=n_prompt,
                negative_original_size=(512, 512),
                negative_target_size=(1024, 1024),
                denoising_end=0.8,
                output_type="latent",
                ).images
    
    image = refiner(
        prompt=new_prompt,
        num_inference_steps=infer_steps,
        denoising_start=0.8,
        image=image,
    ).images[0]
   
    
    
    # save the initial images in the backup folder
    if not os.path.exists(tmp_folder):
        os.makedirs(tmp_folder, exist_ok=True)
    image.save(os.path.join(tmp_folder, image_filename))
    return image


def load_dinov2():
    dinov2_vitl14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14').cuda()
    dinov2_vitl14.eval()
    return dinov2_vitl14


if __name__ == "__main__":
    args = config_2_args("consistent_character/config/erin.yaml")
    _ = train_loop(args, args.max_loop, start_from=0)
    
    print(args)