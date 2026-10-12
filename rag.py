import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import json
import faiss
from openai import OpenAI
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from search_demo import load_index_and_chunks, search

load_dotenv()
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://maas.qianwenaiapi.com/compatible-mode/v1"
)

def build_prompt(query, results):
    context = []
    for i, r in enumerate(results):
        context.append(
            f"[资料{i+1}|来源: {r['source']}\n{r['text']}"
        )
    context_str = "\n\n".join(context)
    return (
        f"基于以下资料回答问题，只使用资料中的信息。"
        f"如果资料中没有相关内容，直接回答“资料中没有找到相关信息”。\n\n"
        f"{context_str}\n\n问题:{query}"
    )

def rag_query(query, index, chunks, model, k=3):
    # 1.检索
    results = search(query, index, chunks, model, k=k)

    # 2.拼Prompt
    prompt = build_prompt(query, results)

    # 3.调用LLM
    try:
        completion = client.chat.completions.create(
            model="qwen3.8-max",
            messages=[{"role": "user", "content": prompt}]
        )
        answer = completion.choices[0].message.content
    except Exception as e:
        print(f"LLM调用错误: {e}")
        return


    print(f"\n问题: {query}")
    print(f"回答: {answer}")
    print("引用来源:")
    for r in results:
        print(f" - {r['source']}(相似度{r['score']:.4f})")

if __name__ == "__main__":
    model = SentenceTransformer("BAAI/bge-small-zh", device="cpu")
    index, chunks = load_index_and_chunks()

    queries = [
        "什么是开放式净值型产品？",
        "理财产品的风险有哪些？",
        "银行理财产品的定义？"
    ]
    for q in queries:
        rag_query(q, index, chunks, model)
