from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from pathlib import Path
import json
from langchain_core.documents import Document

def load_documents(docs_dir):
    result = {}
    for file in Path(docs_dir).rglob("*.md"):
        result[file.name] = file.read_text(encoding="utf-8")
    return result

def split_text(text):
    # 如果文档没有Markdown标题(不含#开头),用普通切块器-返回字符串列表
    if "#" not in text:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=300, chunk_overlap=50,
            separators=["\n\n", "\n", "。", ""]
        )
        chunks = splitter.split_text(text)
        # 包装成Document
        return [Document(page_content=c, metadata={}) for c in chunks]
    # 否则用Markdown切块器-返回Document对象列表
    headers_to_split_on = [
        ("#", "Header1"),
        ("##", "Header2"),
        ("###", "Header3"),
    ]
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False     # 保留标题在块力，方便LLM知道上下文
    )
    return splitter.split_text(text)

def build_chunks(doc_dir):
    docs = load_documents(doc_dir)
    all_chunks = []
    for filename, content in docs.items():
        chunk_of_file = split_text(content)
        for i, chunk in enumerate(chunk_of_file):
            all_chunks.append({
                "text": chunk.page_content,
                "source": filename,
                "chunk_id": i,
                "metadata": chunk.metadata
            })
    return all_chunks

if __name__ == "__main__":
    chunks = build_chunks("docs")
    with open("chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"共 {len(chunks)} 块")