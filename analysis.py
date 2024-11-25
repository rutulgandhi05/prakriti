import os
import re
import torch
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from clip import clip
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms
import seaborn as sns
import tabulate

# Configuration
input_folder = "D:/Project/prakriti/data/3r1n_mod1/loop_images/3r1n/0"  # Folder containing generated images
output_folder = "D:/Project/prakriti/data/3r1n_mod1"  # Folder to save analysis results
reference_image_index = 0  # Index of the reference image for face similarity

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Function to clean file names and extract prompts
def extract_prompt(file_name):
    prompt = re.sub(r"[_\(\)0-9]", " ", file_name)  # Remove numbers, brackets, and underscores
    prompt = prompt.replace(",", "").strip()  # Remove commas and extra spaces
    return prompt

# Load CLIP Model
def load_clip_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess, device

# Calculate CLIP similarity
def calculate_clip_similarity(model, preprocess, device, prompt, image_path):
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    text = clip.tokenize([prompt]).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        similarity = torch.cosine_similarity(image_features, text_features).item()
    return similarity

# Load FaceNet Model
def load_facenet_model():
    return InceptionResnetV1(pretrained="vggface2").eval()

# Extract Face Embedding
def extract_face_embedding(model, image_path):
    image = Image.open(image_path).convert("RGB").resize((160, 160))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5, std=0.5)
    ])
    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        embedding = model(image_tensor)
    return embedding

# Calculate Face Similarity
def calculate_face_similarity(model, reference_embedding, test_image_path):
    test_embedding = extract_face_embedding(model, test_image_path)
    similarity = torch.cosine_similarity(reference_embedding, test_embedding).item()
    return similarity

# Perform Analysis
def analyze_images(folder_path, reference_index, output_folder):
    # List all image files
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    image_files.sort()  # Sort for consistency

    # Extract prompts and file paths
    image_data = [{"path": os.path.join(folder_path, f), "prompt": extract_prompt(f)} for f in image_files]

    # Load models
    clip_model, clip_preprocess, clip_device = load_clip_model()
    facenet_model = load_facenet_model()

    # Use reference image for face similarity
    reference_image_path = image_data[reference_index]['path']
    reference_embedding = extract_face_embedding(facenet_model, reference_image_path)

    # Analysis results
    results = []

    # Analyze each image
    for idx, image_info in enumerate(image_data):
        prompt = image_info['prompt']
        image_path = image_info['path']

        # CLIP Similarity
        clip_similarity = calculate_clip_similarity(clip_model, clip_preprocess, clip_device, prompt, image_path)

        # Face Similarity
        face_similarity = calculate_face_similarity(facenet_model, reference_embedding, image_path)

        # Append results
        results.append({"Prompt": prompt, "Image_Path": image_path, 
                        "CLIP_Similarity": clip_similarity, "Face_Similarity": face_similarity})
        print(f"Processed {idx + 1}/{len(image_data)}: CLIP={clip_similarity:.4f}, Face={face_similarity:.4f}")

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Save analysis results as table
    save_results_as_table(df, output_folder)

    # Generate plots
    generate_plots(df, output_folder)

    print(f"Analysis completed. Results saved to {output_folder}")

# Save Results as Table Image
def save_results_as_table(df, output_folder):
    table_path = os.path.join(output_folder, "analysis_table.png")
    fig, ax = plt.subplots(figsize=(12, len(df) * 0.6))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(len(df.columns))))
    plt.savefig(table_path, bbox_inches='tight')
    print(f"Table saved at {table_path}")

# Generate Plots
def generate_plots(df, output_folder):
    # CLIP Similarity Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Prompt", y="CLIP_Similarity", data=df)
    plt.xticks(rotation=45, ha='right')
    plt.title("CLIP Similarity by Prompt")
    plt.xlabel("Prompt")
    plt.ylabel("CLIP Similarity")
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "clip_similarity_plot.png"))
    print("CLIP similarity plot saved.")

    # Face Similarity Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Prompt", y="Face_Similarity", data=df)
    plt.xticks(rotation=45, ha='right')
    plt.title("Face Similarity by Prompt")
    plt.xlabel("Prompt")
    plt.ylabel("Face Similarity")
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "face_similarity_plot.png"))
    print("Face similarity plot saved.")

# Run Analysis
if __name__ == "__main__":
    analyze_images(input_folder, reference_image_index, output_folder)
