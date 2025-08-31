import os
import json
from typing import List, Optional
from fastapi import FastAPI, HTTPException,Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from dotenv import load_dotenv
import torch
import faiss
import clip
from PIL import Image
from groq import Groq
from transformers import BlipProcessor, BlipForConditionalGeneration


# Load env vars
load_dotenv()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

INDEX_FILE = os.getenv("INDEX_FILE")
META_FILE = os.getenv("META_FILE")
IMAGES_DIR = os.getenv("IMAGES_DIR")
BLIP_MODEL_DIR = os.getenv("BLIP_MODEL_DIR")
TOP_K_DEFAULT = int(os.getenv("TOP_K_DEFAULT", 5))
MAX_TOKENS = int(os.getenv("GROQ_MAX_TOKENS", 512))
GROQ_MODEL = os.getenv("GROQ_MODEL")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# FastAPI setup
app = FastAPI(title="Image Search API")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)
app.mount("/images", StaticFiles(directory=IMAGES_DIR), name="images")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
# Load models & data
faiss_index = faiss.read_index(INDEX_FILE)
with open(META_FILE, "r", encoding="utf-8") as f:
    METADATA: List[dict] = json.load(f)

clip_model, clip_preprocess = clip.load("ViT-B/32", device=DEVICE)
blip_processor = BlipProcessor.from_pretrained(BLIP_MODEL_DIR)
blip_model = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL_DIR).to(DEVICE)
blip_model.eval()

# Helpers
def encode_query(query: str):
    tokens = clip.tokenize([query]).to(DEVICE)
    with torch.no_grad():
        text_emb = clip_model.encode_text(tokens)
    text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
    return text_emb.cpu().numpy().astype("float32")

def generate_caption(image_path: str) -> str:
    image = Image.open(image_path).convert("RGB")
    inputs = blip_processor(images=image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        ids = blip_model.generate(**inputs, max_length=32, num_beams=3)
    return blip_processor.decode(ids[0], skip_special_tokens=True)

def generate_explanation(query: str, caption: str) -> str:
    if not groq_client:
        return f'This image matches "{query}" because: {caption}'
    prompt = (
        f"Query: {query}\n"
        f"Image description: {caption}\n"
        "Explain in 1-2 sentences why this image matches the query.\n"
        "Output format: Plain sentence without numbering and starting with 'matches because'"
    )
    resp = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=MAX_TOKENS,
    )
    return resp.choices[0].message.content.strip()

# Request model
class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = TOP_K_DEFAULT

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/search")
async def search(req: SearchRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    top_k = max(1, min(int(req.top_k or TOP_K_DEFAULT), 10))
    qvec = encode_query(req.query)
    distances, indices = faiss_index.search(qvec, top_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0 or idx >= len(METADATA):
            continue
        meta = METADATA[idx]
        filename = meta.get("filename") or os.path.basename(meta.get("path"))
        image_path = os.path.join(IMAGES_DIR, filename)
        caption = generate_caption(image_path)
        explanation = generate_explanation(req.query, caption)
        results.append({
            "path": f"/images/{filename}",
            "filename": filename,
            "distance": float(dist),
            "caption": caption,
            "explanation": explanation
        })

    return JSONResponse({"query": req.query, "top_k": top_k, "results": results})
