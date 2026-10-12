# 个人知识库 RAG 问答系统

## 项目简介
基于FAISS 和 通义千问API的本地知识库问答系统，支持纯文档与Markdown知识库。

## 技术栈
- Python 3.13
- langchain_text_splitters
- sentence_transformers(BAAI/bge-small-zh)
- faiss-cpu
- 通义千问 API(OpenAI兼容接口)


## 核心流程
docs/ → prepare_chunk_v2.py → chunks.json
                                     ↓
                              build_index.py → faiss.index
                                     ↓
用户提问 → search_demo.py → 相关块 → rag.py → 回答

## 关键设计
**1. 混合切块策略**
- Markdown 文档 → MarkdownHeaderTextSplitter 按标题切
- 非结构化文档 → RecursiveCharacterTextSplitter 按字数切
- 原因：最初统一用字数切，Markdown 文档里不同主题被混成一块，检索时都命中同一个块；改用标题切块后，相似度从 0.72 提升到 0.83

**2. FAISS 向量检索**
- IndexFlatIP 内积索引，归一化向量后等价余弦相似度
- 检索 top-3，注入 Prompt 约束 LLM 只基于资料回答

**3. 向量模型调用**
- HF_HUB_OFFLINE=1 解决模型加载卡住
- device='cpu' 避开 cuDNN 问题

## 效果评估
10个测试问题，准确率100%，相似度0.73-0.89。

> 首轮测试准确率为 90%（“开放式净值型产品”检索失败）。
> 原因是该文档无 Markdown 标题，整份被当成一块，语义被稀释。
> 改用混合切块策略后，相似度从 0.72 提升到 0.83。

## 运行方式
### 1. 安装依赖
```bash
pip install langchain-text-splitters sentence-transformers faiss-cpu openai python-dotenv
```

### 2. 配置APIKey
在项目根目录创建.env文件:
```
DASHSCOPE_API_KEY=你的千问API Key
```

### 3. 按顺序运行
```bash
python prepare_chunk_v2.py    # 切块，生成 chunks.json
python build_index.py         # 向量化，生成 faiss.index
python rag.py                 # 运行问答
```



