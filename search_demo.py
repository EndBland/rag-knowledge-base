import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import json
import faiss
from sentence_transformers import SentenceTransformer

def load_index_and_chunks():
    index = faiss.read_index("faiss.index")

    with open("chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return index, chunks

def search(query, index, chunks, model, k=3):
    query_vector = model.encode([query], normalize_embeddings=True)
    scores, indices = index.search(query_vector, k)
    result = []
    for i, idx in enumerate(indices[0]):
        result.append({
            "text": chunks[idx]["text"],
            "source": chunks[idx]["source"],
            "score": float(scores[0][i])
        })
    return result

if __name__ == "__main__":
    print("1. 开始加载模型")
    model = SentenceTransformer("BAAI/bge-small-zh", device="cpu")
    print("2. 模型加载完成")

    index, chunks = load_index_and_chunks()
    print(f"3. 索引和 chunks 加载完成，共 {len(chunks)} 块")

    queries = [
        "变量的作用是什么？",
        "注释的作用是什么？",
        "Python怎么定义变量？"
    ]
    for q in queries:
        print(f"\n===== 问题：{q} =====")
        results = search(q, index, chunks, model)
        for r in results:
            print(f"--- 相似度 {r['score']:.4f} | {r['source']} ---")
            print(r["text"][:80])
            print()