from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
import json

def load_documents(docs_dir):
    result = {}
    for file in Path(docs_dir).rglob("*.md"):
        result[file.name] = file.read_text(encoding="utf-8")
    return result

def split_text(text, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "```", "\n", "。", "！", "？", "；", ""]
    )
    return splitter.split_text(text)

def build_chunks(doc_dir):
    docs = load_documents(doc_dir)
    all_chunks = []
    for filename, content in docs.items():
        chunk_of_file = split_text(content)
        for i, chunk in enumerate(chunk_of_file):
            all_chunks.append({
                "text": chunk,
                "source": filename,
                "chunk_id": i
            })
    return all_chunks

if __name__ == "__main__":
    chunks = build_chunks("docs")
    with open("chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"共 {len(chunks)} 块")