"""Download all-MiniLM-L6-v2 to local models/ directory."""
import os, sys
from sentence_transformers import SentenceTransformer

model_name = "all-MiniLM-L6-v2"
local_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "models", "all-MiniLM-L6-v2")

print(f"Downloading {model_name} ...")
model = SentenceTransformer(model_name)
print(f"Saving to {local_path} ...")
model.save(local_path)
print("Done! Model saved locally.")
