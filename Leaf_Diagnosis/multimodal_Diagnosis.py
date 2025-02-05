import os
from glob import glob
from PIL import Image
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import ollama

# ===============================================================
# Step 1 — Load image embeddings model 
model = SentenceTransformer('clip-ViT-B-32')

# ===============================================================
# Step 2 — Get image embeddings from dataset images
def generate_clip_embeddings(images_path, model):

    image_paths = glob(os.path.join(images_path, '**/*.JPG'), recursive=True)
    
    embeddings = []
    for img_path in image_paths:
        image = Image.open(img_path)
        embedding = model.encode(image)
        embeddings.append(embedding)
    
    return embeddings, image_paths



# ===============================================================
# Step 3 — Generate FAISS Index
def create_faiss_index(embeddings, image_paths, output_path):

    dimension = len(embeddings[0])
    index = faiss.IndexFlatIP(dimension)
    index = faiss.IndexIDMap(index)
    
    vectors = np.array(embeddings).astype(np.float32)

    # Add vectors to the index with IDs
    index.add_with_ids(vectors, np.array(range(len(embeddings))))
    
    # Save the index
    faiss.write_index(index, output_path)
    print(f"Index created and saved to {output_path}")
    
    # Save image paths
    with open(output_path + '.paths', 'w') as f:
        for img_path in image_paths:
            f.write(img_path + '\n')
    
    return index




# ===============================================================
# Step 4 — Retrieve Images by a Text Query or an Input Image
def retrieve_similar_images(query, model, index, image_paths, top_k=3):
    
    # query preprocess:
    if query.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif', '.JPG')):
        query = Image.open(query)

    query_features = model.encode(query)
    query_features = query_features.astype(np.float32).reshape(1, -1)

    distances, indices = index.search(query_features, top_k)

    retrieved_images = [image_paths[int(idx)] for idx in indices[0]]

    return query, retrieved_images


# ===============================================================
# Step 5 — Visualise results
def visualize_results(query, retrieved_images):
    plt.figure(figsize=(12, 5))

    # If image query
    if isinstance(query, Image.Image):
        plt.subplot(1, len(retrieved_images) + 1, 1)
        plt.imshow(query)
        plt.title("Query Image")
        plt.axis('off')
        start_idx = 2

    # If text query
    else:
        plt.subplot(1, len(retrieved_images) + 1, 1)
        plt.text(0.5, 0.5, f"Query:\n\n '{query}'", fontsize=16, ha='center', va='center')
        plt.axis('off')
        start_idx = 2

    # Display images
    for i, img_path in enumerate(retrieved_images):

        plt.subplot(1, len(retrieved_images) + 1, i + start_idx)
        plt.imshow(Image.open(img_path))
        plt.title(f"Match {i + 1}")
        plt.axis('off')

    plt.show()
    
    
# ===============================================================
# Step 6 — Based on similarity search generate a diagnosis
def generate_description(disease_list):
    prompt = f"""
    I have performed an image similarity search on a dataset of diseased and healthy leaves using a CLIP-based model. 
    
    The search retrieved five images that most closely match an input image, ranked from most similar to least similar.
    
    Each retrieved image has a filename that includes either the name of a plant disease or the label "healthy". The first image is the most similar, 
    making its disease label the most probable match, while the last image is the least similar, making its disease label the least probable.
    Additionally, some diseases may appear more than once among the five retrieved images. The frequency of a disease in this list should also 
    be considered—if a disease appears multiple times, it is more likely to be correct.
    
    Here are the retrieved disease names in order of similarity:
    1. {disease_list[0]} (Most similar)
    2. {disease_list[1]}
    3. {disease_list[2]}
    4. {disease_list[3]}
    5. {disease_list[4]} (Least similar)

   Please analyze the ranking and frequency of occurrences to determine the most probable  label for the input image. 
   Assign higher probability to labels (disease name or healthy) that appear more frequently and are ranked higher. If multiple diseases labels are present, provide reasoning and rank their likelihood.
    """

    response = ollama.chat(model="llama3.2-vision:latest", messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]
    
# ===============================================================================================
DATASET_PATH = 'Dataset'
OUTPUT_INDEX_PATH = "vector.index"
embeddings, image_paths = generate_clip_embeddings(DATASET_PATH, model)
index = create_faiss_index(embeddings, image_paths, OUTPUT_INDEX_PATH)
# ===============================================================================================

in_image_paths = ['leaves\\blight.jpg', 'leaves\\healthy.jpg', 'leaves\\rust.jpg']
for img_path in in_image_paths:
    query = img_path
    print("===============================================================================================")
    print("Diagnosis for image: ", query)    
    query, retrieved_images = retrieve_similar_images(query, model, index, image_paths, 5)
    print(retrieved_images)
    diagnosis=generate_description(retrieved_images)
    print(diagnosis)
    visualize_results(query, retrieved_images)

