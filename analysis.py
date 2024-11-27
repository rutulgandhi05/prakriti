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

# Configuration
input_folder = "D:/Project/prakriti/data/3r1n_final_simple/3r1n/0"  # Folder containing images
output_folder = "D:/Project/prakriti/data/3r1n_final_simple/analysis1"  # Folder to save results
reference_image_index = 0  # Index of reference image

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Extract prompt from file name
def extract_prompt(file_name):
    prompt = re.sub(r"[_\(\)0-9]", " ", file_name)  # Remove unwanted characters
    prompt = prompt.replace(",", "").replace(".png", "").replace("r n", "3r1n").strip()
    return prompt

# Load CLIP model
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

# Load FaceNet model
def load_facenet_model():
    return InceptionResnetV1(pretrained="vggface2").eval()

# Extract face embedding
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

# Calculate face similarity
def calculate_face_similarity(model, reference_embedding, test_image_path):
    test_embedding = extract_face_embedding(model, test_image_path)
    similarity = torch.cosine_similarity(reference_embedding, test_embedding).item()
    return similarity

# Plot histogram/density
def plot_histogram(df, column, title, xlabel, output_path):
    plt.figure(figsize=(10, 6))
    sns.histplot(df[column], kde=True, bins=20, color="skyblue")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# Plot scatter plot with color-coded similarity ranges
def plot_scatter(df, output_path):
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(df["CLIP_Similarity"], df["Face_Similarity"],
                          c=df["CLIP_Similarity"] + df["Face_Similarity"],
                          cmap="viridis", alpha=0.7)
    plt.colorbar(scatter, label="Combined Similarity (CLIP + Face)")
    plt.title("CLIP Similarity vs Face Similarity")
    plt.xlabel("CLIP Similarity")
    plt.ylabel("Face Similarity")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# Plot box plot for similarity distributions
def plot_box(df, output_path):
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df[["CLIP_Similarity", "Face_Similarity"]])
    plt.xticks([0, 1], ["CLIP Similarity", "Face Similarity"])
    plt.title("Distribution of Similarity Scores")
    plt.ylabel("Similarity Score")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# Plot average similarity bar chart
def plot_clustered_bar_chart(df, output_path):
    avg_clip = df["CLIP_Similarity"].mean()
    avg_face = df["Face_Similarity"].mean()
    categories = ["CLIP Similarity", "Face Similarity"]
    scores = [avg_clip, avg_face]

    plt.figure(figsize=(8, 6))
    sns.barplot(x=categories, y=scores, palette="muted")
    plt.title("Average Similarity Scores")
    plt.ylabel("Similarity Score")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# Plot trends over image indices
def plot_line(df, output_path):
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df["CLIP_Similarity"], label="CLIP Similarity", marker="o", linestyle="--")
    plt.plot(df.index, df["Face_Similarity"], label="Face Similarity", marker="o", linestyle="-")
    plt.title("Trends in Similarity Scores Across Images")
    plt.xlabel("Image Index")
    plt.ylabel("Similarity Score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# Perform analysis
def analyze_images(folder_path, reference_index, output_folder):
    # List image files
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    image_files.sort()

    # Extract prompts
    image_data = [{"path": os.path.join(folder_path, f), "prompt": extract_prompt(f)} for f in image_files]

    # Load models
    clip_model, clip_preprocess, clip_device = load_clip_model()
    facenet_model = load_facenet_model()

    # Reference embedding for face similarity
    reference_image_path = image_data[reference_index]['path']
    reference_embedding = extract_face_embedding(facenet_model, reference_image_path)

    # Analyze images
    results = []
    for idx, image_info in enumerate(image_data):
        prompt = image_info['prompt']
        image_path = image_info['path']
        
        # CLIP Similarity
        clip_similarity = calculate_clip_similarity(clip_model, clip_preprocess, clip_device, prompt, image_path)
        
        # Face Similarity
        face_similarity = calculate_face_similarity(facenet_model, reference_embedding, image_path)
        
        # Save results
        results.append({
            "Prompt": prompt,
            "Image_Path": image_path,
            "CLIP_Similarity": clip_similarity,
            "Face_Similarity": face_similarity
        })
        print(f"Processed {idx + 1}/{len(image_data)}: CLIP={clip_similarity:.4f}, Face={face_similarity:.4f}")

    # Convert results to DataFrame
    df = pd.DataFrame(results)

    # Generate plots
    plot_histogram(df, "CLIP_Similarity", "Distribution of CLIP Similarity", "CLIP Similarity",
                   f"{output_folder}/clip_similarity_histogram.png")
    plot_histogram(df, "Face_Similarity", "Distribution of Face Similarity", "Face Similarity",
                   f"{output_folder}/face_similarity_histogram.png")
    plot_scatter(df, f"{output_folder}/scatter_plot.png")
    plot_box(df, f"{output_folder}/boxplot_similarity.png")
    plot_clustered_bar_chart(df, f"{output_folder}/clustered_bar_chart.png")
    plot_line(df, f"{output_folder}/lineplot_similarity.png")
    print(f"Analysis completed. Results and plots saved in {output_folder}.")

    df.to_csv(f"{output_folder}/analysis_results.csv", index=False)
    print(f"Results saved to {output_folder}/analysis_results.csv")

# Run analysis
if __name__ == "__main__":
    analyze_images(input_folder, reference_image_index, output_folder)
