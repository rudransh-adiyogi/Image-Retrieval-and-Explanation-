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


