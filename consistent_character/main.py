import argparse
import yaml
import torch
import numpy as np
import os
import random
from sklearn.cluster import KMeans
from diffusers import StableDiffusionXLPipeline
from hdbscan import HDBSCAN
import sdxl_consistent_character as sdxl  # Importing the fine-tuning module
from transformers import CLIPProcessor, CLIPModel
from facenet_pytorch import InceptionResnetV1

# ------------------------------
# Command-line Argument Parsing
# ------------------------------
def parse_args():
    """
    Parse command-line arguments to get the config file.
    """
    parser = argparse.ArgumentParser(description="Run the full image generation and fine-tuning pipeline.")
    parser.add_argument('--config', type=str, required=True, help="Path to the YAML configuration file.")
    return parser.parse_args()

# ------------------------------
# YAML Configuration Loading
# ------------------------------
def load_config(config_path):
    """
    Load the YAML configuration file.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# ------------------------------
# Image Generation
# ------------------------------
def generate_images_with_fixed_seed(pipe: StableDiffusionXLPipeline, prompt: str, batch_size=8, seed=42, negative_prompt=None):
    """
    Generate a batch of images with a fixed random seed and a negative prompt to avoid unwanted styles.
    """
    generator = torch.manual_seed(seed)
    images = [pipe(prompt=prompt, negative_prompt=negative_prompt, generator=generator).images[0] for _ in range(batch_size)]
    return images

# ------------------------------
# Feature Extraction
# ------------------------------
def extract_combined_features(image):
    """
    Extract features from multiple feature extractors (DINOv2, CLIP, Facenet).
    Combine these features for better clustering performance.
    """
    # Initialize the feature extractors (CLIP, Facenet)
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    facenet = InceptionResnetV1(pretrained='vggface2').eval()

    # Convert the image to tensor for feature extraction
    image_tensor = torch.tensor(np.array(image)).permute(2, 0, 1).unsqueeze(0).float() / 255.0

    # Extract CLIP features
    clip_inputs = clip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        clip_features = clip_model.get_image_features(**clip_inputs).squeeze().cpu().numpy()

    # Extract Facenet features (for facial recognition features)
    facenet_features = facenet(image_tensor).detach().cpu().numpy().squeeze()

    # Combine features into a single feature vector
    combined_features = np.concatenate([clip_features, facenet_features], axis=0)
    return combined_features

# ------------------------------
# Clustering
# ------------------------------
def kmeans_clustering_with_adaptive_threshold(args, data_points, images=None):
    """
    Perform K-means clustering and adjust the similarity threshold based on variability of distances.
    """
    kmeans = KMeans(n_clusters=args['kmeans_center'], init='k-means++', random_state=42)
    kmeans.fit(data_points)
    labels = kmeans.labels_

    clusters = {i: [] for i in range(args['kmeans_center'])}
    for idx, label in enumerate(labels):
        clusters[label].append((data_points[idx], images[idx])) if images else None

    # Filter small clusters based on minimum cluster size (dmin_c)
    selected_clusters = [cluster for cluster in clusters.values() if len(cluster) >= args['dmin_c']]

    # Dynamically adjust the threshold for clustering
    avg_distance = np.mean([np.linalg.norm(e[0] - np.mean([c[0] for c in cluster], axis=0)) for cluster in selected_clusters for e in cluster])
    threshold = args['convergence_scale'] + avg_distance * 0.5  # Adjust dynamically based on variability

    most_cohesive_cluster = min(selected_clusters, key=lambda cluster: 
                                np.mean([np.linalg.norm(e[0] - np.mean([c[0] for c in cluster], axis=0)) 
                                         for e in cluster]))

    return most_cohesive_cluster

# ------------------------------
# Latent Space Interpolation
# ------------------------------
def interpolate_in_latent_space(latent1, latent2, alpha=0.5):
    """
    Perform interpolation between two latent representations to smooth out differences.
    """
    return latent1 * (1 - alpha) + latent2 * alpha

# ------------------------------
# Saving Images
# ------------------------------
def save_images(images, output_dir, prefix='interpolated'):
    """
    Save the generated or interpolated images to a specified output directory.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for i, image in enumerate(images):
        image.save(os.path.join(output_dir, f"{prefix}_{i}.png"))

# ------------------------------
# Training Loop (Image Generation + Clustering)
# ------------------------------
def train_loop(pipe, args):
    """
    Main training loop with batch generation, clustering, interpolation, and saving fine-tuned images.
    This function returns the interpolated images for further fine-tuning.
    """
    all_interpolated_images = []

    for loop in range(args['loop_num']):
        print(f"Starting loop {loop + 1}/{args['loop_num']}")
        
        # Generate a batch of images with fixed seed and negative prompt
        images = generate_images_with_fixed_seed(pipe, prompt=args['inference_prompt'], batch_size=args['batch_size'], seed=random.randint(0, 10000), negative_prompt=args['negative_prompt'])
        print(f"Generated {len(images)} images.")

        # Extract combined embeddings (DINOv2 + CLIP + Facenet)
        embeddings = [extract_combined_features(image) for image in images]
        
        # Perform adaptive clustering
        most_cohesive_cluster = kmeans_clustering_with_adaptive_threshold(args, embeddings, images)
        print(f"Most cohesive cluster selected with {len(most_cohesive_cluster)} images.")

        # Perform latent space interpolation between the selected images for consistency
        interpolated_images = []
        for i in range(0, len(most_cohesive_cluster) - 1, 2):
            latent1 = pipe.encode_image(most_cohesive_cluster[i][1])
            latent2 = pipe.encode_image(most_cohesive_cluster[i + 1][1])
            interpolated_latent = interpolate_in_latent_space(latent1, latent2)
            interpolated_images.append(pipe.decode_image(interpolated_latent))
        print(f"Interpolated {len(interpolated_images)} images.")

        # Save interpolated images
        output_dir = os.path.join(args['output_dir'], f"loop_{loop + 1}")
        save_images(interpolated_images, output_dir, prefix='interpolated')
        print(f"Saved interpolated images to {output_dir}.")

        # Append interpolated images to return later for fine-tuning
        all_interpolated_images.extend(interpolated_images)

        # Save model checkpoint (if necessary)
        checkpoint_path = os.path.join(output_dir, "model_checkpoint.pt")
        torch.save(pipe.state_dict(), checkpoint_path)
        print(f"Saved model checkpoint to {checkpoint_path}.")

    print("Training loop complete.")
    return all_interpolated_images

# ------------------------------
# Main Function
# ------------------------------
if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_args()
    
    # Load configuration file
    config = load_config(args.config)

    # Initialize the Stable Diffusion pipeline
    pipe = StableDiffusionXLPipeline.from_pretrained(config['internal_model_path'], torch_dtype=torch.float16)

    # Run the image generation and clustering
    interpolated_images = train_loop(pipe, config)

    # After image generation, invoke the fine-tuning using `sdxl_the_chosen_one`
    print("Starting fine-tuning with LoRA and textual inversion...")
    
    # Invoke textual inversion and fine-tuning from sdxl_the_chosen_one.py
    sdxl.textual_inversion(pipe, config)  # Fine-tune with textual inversion
    sdxl.train_pipeline(pipe, interpolated_images, config)  # Perform LoRA fine-tuning
