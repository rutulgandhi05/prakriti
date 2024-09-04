import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from diffusers import StableDiffusionPipeline, DiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer
from torchvision import transforms
from sklearn.cluster import KMeans
from PIL import Image
from tqdm import tqdm

# Initialize the Stable Diffusion model
model_id = "stabilityai/stable-diffusion-xl-base-1.0"
pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
pipe = pipe.to("cuda")

# Initialize CLIP model for text embeddings
clip_model = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")


# Custom text embeddings
custom_embeddings = torch.randn((1, clip_model.config.hidden_size), requires_grad=True, device="cuda")

# Transformation for the images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def generate_images(prompt, num_images, seed=42, save_dir="generated_images", iteration=0):
    os.makedirs(save_dir, exist_ok=True)
    generator = torch.manual_seed(seed)
    images = []
    for i in range(num_images):
        image = pipe(prompt, generator=generator).images[0]
        images.append(image)
        image.save(f"{save_dir}/image_{iteration}_{i}.png")
    return images


def embed_images(images):
    embeddings = []
    for image in images:
        img_tensor = transform(image).unsqueeze(0).to("cuda")
        with torch.no_grad():
            embedding = clip_model.get_image_features(img_tensor)
        embeddings.append(embedding.cpu().numpy())
    return np.array(embeddings)


class LoRALayer(nn.Module):
    def __init__(self, input_dim, rank):
        super(LoRALayer, self).__init__()
        self.down = nn.Linear(input_dim, rank, bias=False)
        self.up = nn.Linear(rank, input_dim, bias=False)
        
    def forward(self, x):
        return self.up(self.down(x))
    

def apply_lora(model, rank):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            lora_layer = LoRALayer(module.in_features, rank).to(module.weight.device)
            lora_layer.up.weight.data = module.weight.data.clone()
            module.weight.data = lora_layer.forward(module.weight).data
            setattr(model, name, lora_layer)



def cluster_embeddings(embeddings, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    clusters = kmeans.fit_predict(embeddings)
    return clusters, kmeans



def fine_tune_model(clusters, images, custom_embeddings, num_epochs=10, lr=1e-4):
    # Prepare dataset
    cluster_images = [images[i] for i in range(len(images)) if clusters[i] == clusters[0]]
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    cluster_tensors = torch.stack([transform(img).to("cuda") for img in cluster_images])
    dataset = torch.utils.data.TensorDataset(cluster_tensors)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Apply LoRA to the model
    apply_lora(pipe.unet, rank=4)
    
    # Optimizer
    optimizer = optim.Adam(pipe.unet.parameters(), lr=lr)
    
    # Training loop
    pipe.unet.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs,) in enumerate(tqdm(dataloader)):
            optimizer.zero_grad()
            outputs = pipe.unet(inputs)
            loss = nn.functional.mse_loss(outputs, inputs)  # Assuming autoencoder-like training
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader)}")
    print("Fine-tuning completed.")


def calculate_average_pairwise_distance(embeddings):
    num_embeddings = len(embeddings)
    distances = []
    for i in range(num_embeddings):
        for j in range(i + 1, num_embeddings):
            distance = np.linalg.norm(embeddings[i] - embeddings[j])
            distances.append(distance)
    return np.mean(distances)


def display_images(images, iteration):
    fig, axes = plt.subplots(1, len(images), figsize=(20, 5))
    for i, image in enumerate(images):
        axes[i].imshow(image)
        axes[i].axis('off')
    plt.suptitle(f"Iteration {iteration}")
    plt.show()


def iterative_refinement(prompt, num_iterations=10, num_images=10, num_clusters=5, convergence_threshold=0.1):
    previous_average_distance = float('inf')
    for i in range(num_iterations):
        print(f"Iteration {i+1}/{num_iterations}")
        images = generate_images(prompt, num_images, iteration=i)
        embeddings = embed_images(images)
        average_distance = calculate_average_pairwise_distance(embeddings)
        print(f"Average pairwise distance: {average_distance}")
        
        display_images(images[:5], i)  # Display the first 5 images
        
        if average_distance < convergence_threshold or abs(previous_average_distance - average_distance) < convergence_threshold:
            print("Convergence criterion met. Stopping iterations.")
            break
        
        previous_average_distance = average_distance
        clusters, kmeans = cluster_embeddings(embeddings, num_clusters)
        fine_tune_model(clusters, images, custom_embeddings)

# Example usage
prompt = "a cute baby cat with big eyes"
iterative_refinement(prompt)