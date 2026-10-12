from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-small-zh", device="cpu")
texts = ["操作系统是什么", "Python怎么定义变量", "理财产品的风险"]
vectors = model.encode(texts, normalize_embeddings=True)

print(vectors.shape)
print(vectors[0][:5])