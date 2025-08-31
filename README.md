# Image-Retrieval-and-Explanation-
This project implements an end-to-end image retrieval and explanation pipeline using CLIP, FAISS, BLIP, and Groq LLM. The system allows users to query with natural language, retrieve the most relevant images, generate captions, and provide explanations through a FastAPI backend with a simple HTML+JS frontend

## 📂 Project Structure
'''
📦 project-root

├── 📂 app # FastAPI backend + frontend (HTML+JS)

 └── 📂 models # Pre-trained models (CLIP, BLIP, etc.)
 
 └── 📄 image_index.faiss # Generated FAISS index
 
 └──📄 metadata.json # Image metadata (id, image path)
 
 └── 📄 requirements.txt # Python dependencies
 
 └── 📄 Dockerfile # Container definition
 
 └── 📂 images # Image dataset

├── 📂 preprocessing # Scripts for dataset download & embedding generation
'''
## Architecture
The system consists of two main pipelines:
1. Preprocessing Pipeline
   •	CLIP Encoder: Encodes text queries into embeddings.
   •	FAISS Index: Stores image embeddings for fast nearest neighbor search.
   •	Metadata Store: Maintains mapping of image IDs and paths for retrieval.
   <img width="728" height="251" alt="image" src="https://github.com/user-attachments/assets/fdc22918-f156-46dd-b52b-6e98f6dbc3d0" />

2. Image Retrieval & Explanation Flow
   <img width="644" height="448" alt="image" src="https://github.com/user-attachments/assets/6f138b1b-d476-4d17-a4c5-d82802b4465c" />
   
   1.	User enters a query on the web UI.
   2.	Frontend sends a POST /search request with JSON {query, top_k}.
   3.	FastAPI encodes the query with CLIP and retrieves top-K matches from FAISS.
   4.	BLIP generates captions for each image.
   5.	Groq LLM provides explanations for why the image matches.
   6.	API responds with:{"results": [ {"image_path": "...","caption": "...","explanation": "..."} ] }
   7.	Frontend renders results as image cards.







