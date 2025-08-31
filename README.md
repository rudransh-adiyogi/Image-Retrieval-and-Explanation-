# Image-Retrieval-and-Explanation-
This project implements an end-to-end image retrieval and explanation pipeline using CLIP, FAISS, BLIP, and Groq LLM. The system allows users to query with natural language, retrieve the most relevant images, generate captions, and provide explanations through a FastAPI backend with a simple HTML+JS frontend

## 📂 Project Structure

📦 project-root

├── 📂 app # FastAPI backend + frontend (HTML+JS)

      └── 📂 models # Pre-trained models (CLIP, BLIP, etc.)
 
      └── 📄 image_index.faiss # Generated FAISS index
 
      └──📄 metadata.json # Image metadata (id, image path)
 
      └── 📄 requirements.txt # Python dependencies
 
      └── 📄 Dockerfile # Container definition
 
      └── 📂 images # Image dataset

├── 📂 preprocessing # Scripts for dataset download & embedding generation

## Architecture
The system consists of two main pipelines:

 ### Preprocessing Pipeline

   •	CLIP Encoder: Encodes text queries into embeddings.

   •	FAISS Index: Stores image embeddings for fast nearest neighbor search.

   •	Metadata Store: Maintains mapping of image IDs and paths for retrieval.

   <img width="728" height="251" alt="image" src="https://github.com/user-attachments/assets/fdc22918-f156-46dd-b52b-6e98f6dbc3d0" />
   
 
 ### Image Retrieval & Explanation Flow

   
   <img width="644" height="448" alt="image" src="https://github.com/user-attachments/assets/6f138b1b-d476-4d17-a4c5-d82802b4465c" />
   
   1.	User enters a query on the web UI.
   2.	Frontend sends a POST /search request with JSON {query, top_k}.
   3.	FastAPI encodes the query with CLIP and retrieves top-K matches from FAISS.
   4.	BLIP generates captions for each image.
   5.	Groq LLM provides explanations for why the image matches.
   6.	API responds with:{"results": [ {"image_path": "...","caption": "...","explanation": "..."} ] }
   7.	Frontend renders results as image cards.

## How to Run

 ### Preprocessing
 
  Create the image dataset. 
 
  #### Prerequisite: Conda environment
    
    1. Set the Environment
            
            cd preprocessing
            
            conda create -n imgsearch python=3.9
            
            conda activate imgsearch
            
            pip install -r requirements.txt

     
     2. Run image download script:
             
             python download_images.py #This downloads images to /app/images/ folder
     
     3. Generate FAISS index & metadata:
             
             python generate_embedding_n_metadata.py #This will create the embeddings and store themin vector DB and metadata.json

 ### Image Retrieval & Explanation

      #### Prerequisite: Docker Desktop or DockerHUB
            
            1.Download Blip model files from **https://huggingface.co/Salesforce/blip-image-captioning-base/tree/main** and save to  /app/models/blip-captioning-model 
            
            2. Go to **https://console.groq.com/keys** and login and get an API key
            
            3. Add it in .env under GROQ_API_KEY.
            
            4. Build the image:
                        docker build -t image-search:latest .
            
            5. Run the container:
                        docker run -p 8000:8000 --env-file .env \
                        -v $PWD/image_index.faiss:/app/image_index.faiss \
                        -v $PWD/metadata.json:/app/metadata.json \
                        -v $PWD/models:/app/models  image-search:latest

            6. Open http://localhost:8000 in your browser.

## Results 

<img width="774" height="385" alt="image" src="https://github.com/user-attachments/assets/1f9d97d3-cd42-4a83-a53d-ec7255de8abd" />

<img width="734" height="425" alt="image" src="https://github.com/user-attachments/assets/2630d3ed-5a09-4cda-b774-807eec9403a5" />

<img width="743" height="384" alt="image" src="https://github.com/user-attachments/assets/bd28b22c-799b-4423-9233-c8254407c84b" />

<img width="667" height="407" alt="image" src="https://github.com/user-attachments/assets/952e9802-8e8f-4afe-bf2f-fee2009abe61" />













