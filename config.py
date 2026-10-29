import os

from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.google_genai import GoogleGenAI

load_dotenv()

llm = GoogleGenAI(
    model="gemini-2.5-flash",  # check AI Studio for a current model name
    api_key=os.getenv("GEMINI_API_KEY"),
)
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")  # local, free

# Global defaults, so nothing silently falls back to OpenAI
Settings.llm = llm
Settings.embed_model = embed_model
