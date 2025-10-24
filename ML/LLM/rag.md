
RAG, retrieval-augmented generation; 检索增强生成。

相当于连接特定知识数据库，对局域知识的索引检索，使其能更好处理问题



## 技术栈

</br>

实现方法：
1. [langchain](https://github.com/langchain-ai/langchain)，需要开发的较多
2. [ragflow](https://github.com/infiniflow/ragflow)， 通用 rag 框架
3. [Langchain-Chatchat](https://github.com/chatchat-space/Langchain-Chatchat)，通用 rag 框架， 看着好久没更新了


</br>

关键模块：
- 文本处理，text_split
- 构建存储向量，embedding，vector db
- 构建索引







## langchain demo


说明：装环境(sagemaker 运行的)， 后面可以 gpt 直接开发（有些 gpt 输出的包过时了， 仅留档）

参考官方文档：https://docs.langchain.com/oss/python/langchain/rag#rag-agents

```bash
# 使用 base 环境 python 3.12.11 torch: 2.6.0+cu124

source activate

pip install --upgrade pip
conda update -n base -c conda-forge conda -y
# 必须，不然后面会报错


conda install -c conda-forge pyarrow -y
conda install -c conda-forge tiktoken -y

pip install langchain langchain-text-splitters langchain-community langchain_openai
pip install pypdf

pip install sentence-transformers
conda install -c pytorch faiss-gpu -y

python -m ipykernel install --user --name=llm_kernel
```


### 1. data load

```python
docs = []
docs_vis = set()

from langchain_community.document_loaders import PyPDFLoader

pdf_paths = [
    "flow_cn.pdf"
]

def docs_extend(pdf_paths):
    for path in pdf_paths:
        if path in docs_vis:
            print(f"{path} 已存在")
            continue
        docs_vis.add(path)
        loader = PyPDFLoader(path)
        docs.extend(loader.load())


docs_extend(pdf_paths)

print(f"✅ 追加 docs {len(docs)} 个文档片段")

```

```python
import re
def clean_pdf_text(text):
    """清理 PDF 文本中的换行符和噪声"""
    
    # 0. 首先规范化所有类型的空格字符
    # 将各种空白字符（包括全角空格、不间断空格等）统一转换为普通空格
    text = re.sub(r'[\u2000-\u200F\u2028-\u202F\u205F\u3000\u00A0\uFEFF]', ' ', text)
    
    # 1. 处理中文换行问题 - 移除中文字符间的换行
    text = re.sub(r'([\u4e00-\u9fff])\n([\u4e00-\u9fff])', r'\1\2', text)
    
    # 2. 处理英文换行问题 - 保留单词间空格
    text = re.sub(r'([a-zA-Z])\n([a-zA-Z])', r'\1 \2', text)
    
    # 3. 移除多余的空白字符
    text = re.sub(r'\n+', ' ', text)  # 多个换行替换为空格
    text = re.sub(r'\s+', ' ', text)  # 多个空格替换为单个空格
    
    # 4. 移除常见的页脚信息
    text = re.sub(r'第\s*\d+\s*页', '', text)  # 中文页码
    text = re.sub(r'Page\s*\d+', '', text)     # 英文页码
    text = re.sub(r'\d+\s*$', '', text)        # 行末数字
    text = re.sub(r'For More Visit : www.LearnEngineering.in', '', text)
    text = re.sub(r'For More Visit :', '', text)
    
    # 5. 清理首尾空白
    text = text.strip()
    
    return text


clean_pdf_text(docs[300].page_content)

```

清理文本并拆分：

```bash
docs_new = []

for doc in docs:
    doc.page_content = clean_pdf_text(doc.page_content)
    docs_new.append(doc)


from langchain_text_splitters import RecursiveCharacterTextSplitter


# 3️⃣ 拆分文本为小块（便于向量化）
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # chunk size (characters)
    chunk_overlap=200,  # chunk overlap (characters)
    add_start_index=True,  # track index in original document
)
all_splits = text_splitter.split_documents(docs_new)

print(f"Split docs into {len(all_splits)} sub-documents.")
```




### 2. embedding & vector store

先 `hf auth login` 添加 token，然后可以下载模型

```python
from langchain_community.embeddings import HuggingFaceEmbeddings

# 使用 LangChain 的 HuggingFace embeddings 包装器
embeddings = HuggingFaceEmbeddings(
    model_name="google/embeddinggemma-300m",
    # model_kwargs={'device': 'cpu'},  # 或 'cuda' 如果有GPU
    encode_kwargs={'normalize_embeddings': True}  # 标准化向量
)

print("✅ LangChain 兼容的 embedding 模型加载成功")
device = embeddings.client.device
print(f"📱 HuggingFaceEmbeddings device: {device}")
```

注意索引在 from_documents 建立了，**会将所有文本块的向量，连同它们的元数据（如来源文件、页码、章节标题等），存储到向量数据库中**。 同时可选 关键词索引（为文本块构建一个传统的倒排索引）

而后面 retrieve_context 称为检索， 使用后 向量相似度搜索，关键词搜索等

```python
# way2. 使用 FAISS 向量存储

from langchain_community.vectorstores import FAISS

# 创建 FAISS 向量存储并添加文档
vector_store = FAISS.from_documents(documents=all_splits, embedding=embeddings)

print(f"✅ 已将 {len(all_splits)} 个文档添加到 FAISS 向量存储")

# 可选：保存 FAISS 索引到本地文件
vector_store.save_local("faiss_index")
print("✅ FAISS 索引已保存到本地")

print(f"📊 FAISS 索引统计:")
print(f"   - 文档数量: {vector_store.index.ntotal}")
print(f"   - 向量维度: {vector_store.index.d}")
```





### 3. tools & agent

```python
## load local vector_store 

from langchain_community.vectorstores import FAISS

try:
    # 尝试加载已存在的索引
    vector_store = FAISS.load_local(
        "faiss_index", 
        embeddings, 
        allow_dangerous_deserialization=True
    )
    print("✅ 从本地加载 FAISS 索引成功")
except:
    # 如果没有已保存的索引，则重新创建
    vector_store = FAISS.from_documents(documents=all_splits, embedding=embeddings)
    vector_store.save_local("faiss_index")
    print("✅ 创建新的 FAISS 索引并保存到本地")

from langchain.tools import tool

@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve information to help answer a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = ("\n" + "*" * 50 + "\n").join(
        (f"Source: {doc.metadata}\n\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs


```


```python
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
import os

os.environ["OPENAI_API_KEY"] = "sk-..."
os.environ["OPENAI_API_BASE"] = "https://api.deepseek.com/v1"


# ✅ 使用 DeepSeek API
llm = ChatOpenAI(
    model="deepseek-chat",  # DeepSeek 的聊天模型名
    # openai_api_key="your_deepseek_api_key_here",
    # openai_api_base="https://api.deepseek.com/v1"
)

# 简单对话
# response = llm.invoke("你好，请用一句话介绍一下你自己。")
# print(response.content)



tools = [retrieve_context]


# If desired, specify custom instructions
prompt = (
"""
"You have access to a tool that retrieves context from all books. "
"Use the tool to help answer user queries."
"""
)

agent = create_agent(
    llm, 
    tools, 
    system_prompt=prompt
)


query = (
    "。。。\n\n"
    "Once you get the answer, look up common extensions of that method."
)

def ask(query):
    for event in agent.stream(
        {"messages": [{"role": "user", "content": query}]},
        stream_mode="values",
    ):
        event["messages"][-1].pretty_print()


ask(query)
```


--------------

参考资料：
- [rag 综述](https://zhuanlan.zhihu.com/p/683651359)
- gpt