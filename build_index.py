import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import json
import sys
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

def load_chunks(path="chunks.json"):
    with open(path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            sys.exit("文件损坏，读取错误")

def build_index(chunks, model):
    texts = [chunk["text"] for chunk in chunks]
    vectors = model.encode(texts, normalize_embeddings=True)
    if vectors.dtype == np.float64:
        vectors = vectors.astype("float32")
    index = faiss.IndexFlatIP(512)
    index.add(vectors)
    return index

if __name__ == "__main__":
    model = SentenceTransformer("BAAI/bge-small-zh", device="cpu")
    chunks = load_chunks()
    index = build_index(chunks, model)
    print(f"索引条数：{index.ntotal}")

    faiss.write_index(index, "faiss.index")
    print("索引已保存到faiss.index")