import os
import shutil
import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from torchvision import transforms as T
from torchvision.transforms.functional import center_crop
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from the_chosen_one import fine_tune_model, config_2_args
from diffusers import StableDiffusionXLPipeline

# Load CLIP model for character consistency check
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").cuda()
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


def compute_character_consistency(dino_model, clip_model, clip_processor, images):
    """
    Compute character consistency by focusing on the character's features.
    """
    # Step 1: Extract character region (center crop or use more advanced segmentation)
    character_images = [extract_character(image) for image in images]

    # Step 2: Use DINOv2 to extract character features
    dino_embeddings = [infer_model(dino_model, image).detach().cpu().numpy() for image in character_images]

    # Step 3: Compare character features using DINOv2 embeddings
    dino_similarity_matrix = cosine_similarity(dino_embeddings)

    # Step 4: Use CLIP for character identity similarity
    inputs = clip_processor(images=character_images, return_tensors="pt", padding=True).to('cuda')
    with torch.no_grad():
        image_features = clip_model.get_image_features(**inputs)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        clip_similarity_matrix = cosine_similarity(image_features.cpu().numpy())

    # Combine DINOv2 and CLIP similarities for character consistency
    combined_similarity = (0.5 * dino_similarity_matrix.mean()) + (0.5 * clip_similarity_matrix.mean())

    return combined_similarity


def extract_character(image):
    """
    Extract the character from the image using center cropping.
    """
    # Simple center crop (assuming the character is in the center)
    width, height = image.size
    crop_size = min(width, height)
    image = center_crop(image, crop_size)
    return image


def kmeans_clustering(args, data_points, images=None):
    """
    Perform KMeans clustering on the DINOv2 embeddings.
    """
    kmeans = KMeans(n_clusters=args.kmeans_center, init='k-means++', random_state=42)
    kmeans.fit(data_points)
    labels = kmeans.labels_

    centers = kmeans.cluster_centers_
    selected_labels = [label for label in labels]
    selected_elements = np.array([data_points[i] for i in range(len(labels))])

    selected_images = [images[i] for i in range(len(labels))] if images else None

    return centers, np.array(selected_labels), selected_elements, selected_images


def load_trained_pipeline(model_path=None, load_lora=False, lora_path=None):
    """
    Load the Stable Diffusion pipeline, optionally with LoRA fine-tuned weights.
    """
    if model_path is not None:
        pipe = StableDiffusionXLPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
        if load_lora and lora_path:
            pipe.load_lora_weights(lora_path)
    else:
        pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")
    pipe.to("cuda")
    return pipe


def infer_model(model, image):
    """
    Infer embeddings from DINOv2 model.
    """
    transform = T.Compose([
        T.Resize((518, 518)),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.228, 0.224, 0.225))
    ])
    image = transform(image).unsqueeze(0).cuda()
    return model(image, is_training=False)


def generate_images(pipe, prompt, infer_steps, index=None):
    """
    Generate a single image using the diffusion model pipeline.
    """
    negative_prompt = "cartoon, anime, sketch, 3d render, unrealistic, painting, blur, distortion"

    # Handle seed for consistent generation if index is provided
    if index is not None:
        torch.manual_seed(index * np.random.randint(1000))

    # Generate the image using the diffusion pipeline
    image = pipe(prompt=prompt, num_inference_steps=infer_steps, guidance_scale=7.5,
                 negative_prompt=negative_prompt).images[0]

    return image


def save_image(image, output_dir, image_index):
    """
    Save the generated image to the specified directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    image_path = os.path.join(output_dir, f"{image_index}.png")
    image.save(image_path)
    print(f"Saved image to {image_path}")


def load_dinov2():
    """
    Load the DINOv2 model from Facebook's repository.
    """
    return torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14').cuda()


def train_loop(args, loop_num: int, vis=True, start_from=0):
    """
    Main training loop that generates images, clusters them with DINOv2,
    and checks convergence for character consistency.
    """
    output_dir_base = args.output_dir
    train_data_dir_base = args.train_data_dir
    num_train_epochs = args.num_train_epochs
    checkpointing_steps = args.checkpointing_steps

    args.kmeans_center = int(args.num_of_generated_img / args.dsize_c)
    init_dist = 0

    for loop in range(start_from, loop_num):
        print(f"Starting loop {loop}/{loop_num - 1}")

        dinov2 = load_dinov2()  # Load DINOv2 model

        # Load diffusion model for image generation
        if loop == 0:
            pipe = load_trained_pipeline()
        else:
            pipe = load_trained_pipeline(
                model_path=os.path.join(output_dir_base, args.character_name, str(loop - 1)),
                load_lora=True,
                lora_path=os.path.join(output_dir_base, args.character_name, str(loop - 1),
                                       f"checkpoint-{checkpointing_steps * num_train_epochs}")
            )

        args.output_dir_per_loop = os.path.join(output_dir_base, args.character_name, str(loop))
        args.train_data_dir_per_loop = os.path.join(train_data_dir_base, args.character_name, str(loop))

        if os.path.exists(args.train_data_dir_per_loop):
            shutil.rmtree(args.train_data_dir_per_loop)
        os.makedirs(args.train_data_dir_per_loop, exist_ok=True)

        # Generate images and extract embeddings with DINOv2
        image_embs = []
        images = []

        # Generation and saving process during image creation
        for n_img in range(args.num_of_generated_img):
            print(f"Generating image {n_img + 1}/{args.num_of_generated_img}")
            # Generate the image using the generate_images function
            image = generate_images(pipe, prompt=args.inference_prompt, infer_steps=args.infer_steps, index=n_img)

            tmp_folder = f"{args.backup_data_dir_root}/{args.character_name}/{loop}"
            if not os.path.exists(tmp_folder):
                os.makedirs(tmp_folder, exist_ok=True)
            # Save the generated image and create a backup
            save_image(image, tmp_folder, n_img + 1)

            images.append(image)
            # Extract embeddings using DINOv2
            image_embs.append(infer_model(dinov2, image).detach().cpu().numpy())

        del pipe
        del dinov2
        torch.cuda.empty_cache()

        # Reshape DINOv2 embeddings and perform clustering
        embeddings = np.array(image_embs).reshape(len(image_embs), -1)
        centers, labels, elements, images = kmeans_clustering(args, embeddings, images=images)

        # Identify the most cohesive cluster
        cohesions = [sum(np.linalg.norm(elements[labels == i] - centers[i], axis=1)) for i in range(len(centers))]
        min_cohesion_label = np.argmin(cohesions)
        idx = np.where(labels == min_cohesion_label)[0]

        cohesive_cluster_images = [images[sample_id] for sample_id in idx]
        for i, sample_id in enumerate(idx):
            cohesive_cluster_images[i].save(os.path.join(args.train_data_dir_per_loop, f"image_{sample_id + 1}.png"))

        # Calculate combined similarity for character consistency
        character_consistency = compute_character_consistency(load_dinov2(), clip_model, clip_processor, cohesive_cluster_images)

        # Check if the character consistency meets the threshold for convergence
        print(f"Character Consistency for loop {loop}: {character_consistency}")

        if character_consistency > args.character_consistency_threshold:
            print(f"Converged based on character consistency at loop {loop}.")
            break  # Exit the loop if convergence is reached

        # Fine-tune the model with the cohesive cluster
        args.character_consistency = character_consistency
        fine_tune_model(args, loop)

        print(f"[{loop}/{loop_num-1}] Finish.")
        torch.cuda.empty_cache()


if __name__ == "__main__":
    args = config_2_args("thechosenone/config/captain.yaml")
    train_loop(args, args.max_loop)
