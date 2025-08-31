import os
import json
import numpy as np
from PIL import Image
import torch
import clip
import faiss


# ---------------------------
# Config
# ---------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_DIR = "./app/images/"              # Path to your downloaded images
INDEX_FILE = "./app/image_index.faiss"  # Vector DB file
META_FILE = "./app/metadata.json"       # Metadata file
LIMIT = 1200                      # Limit number of images to process (set None for all)


# ---------------------------
# Load CLIP model
# ---------------------------
print("Loading CLIP model...")
model, preprocess = clip.load("ViT-B/32", device=device)


# ---------------------------
# Function to get embedding
# ---------------------------
def get_image_embedding(image_path):
    """Returns normalized CLIP embedding for given image."""
    image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model.encode_image(image)
    embedding = embedding / embedding.norm(dim=-1, keepdim=True)  # Normalize
    return embedding.cpu().numpy().astype("float32")


def build_vector_db(image_dir=IMAGE_DIR, index_file=INDEX_FILE, meta_file=META_FILE, limit=None):
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('png','jpg','jpeg','webp'))]
    if limit:
        image_files = image_files[:limit]
    print(f"Found {len(image_files)} images to process.")

    embeddings = []
    metadata = []

    for idx, img_file in enumerate(image_files, start=1):
        img_path = os.path.join(image_dir, img_file)
        img_path=img_path.replace('\\','/')
        try:
            emb = get_image_embedding(img_path)
            embeddings.append(emb[0])
            metadata.append({
                "filename": img_file,
                "path": img_path
            })
            if idx % 50 == 0:
                print(f"Processed {idx}/{len(image_files)} images...")
        except Exception as e:
            print(f"Skipping {img_file} due to error: {e}")

    embeddings_np = np.array(embeddings, dtype="float32")
    dim = embeddings_np.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings_np)

    faiss.write_index(index, index_file)
    with open(meta_file, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"✅ Saved {len(metadata)} embeddings to '{index_file}' and metadata to '{meta_file}'.")