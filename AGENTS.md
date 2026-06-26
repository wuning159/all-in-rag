# AGENTS.md

## 项目背景

这个项目基于 Datawhale `all-in-rag` 学习资料，用于系统学习 RAG，并逐步沉淀一个可演示、可扩展的业务智能问答系统。

当前学习主线围绕 RAG MVP 的四个阶段展开：

1. 数据准备与清洗
2. 索引构建
3. 检索策略优化
4. 生成与提示工程

项目当前不仅是课程代码练习，也承担个人学习笔记、环境验证和后续业务 demo 原型的作用。

## 用户是谁

用户是本项目的学习和实践负责人，正在系统学习 RAG，并希望把学习过程中的关键理解、工程判断、环境问题和业务方案沉淀下来。

用户的工作方式偏实践驱动：

- 先跟着课程跑通示例。
- 遇到问题时定位根因。
- 把重要讨论归纳进学习笔记。
- 最终希望形成可演示的业务智能问答 demo。

后续不管在公司还是家里打开这个项目，助手都应该优先读取本文件和学习笔记，快速恢复上下文。

## 当前业务目标

当前重点业务实践目标是：围绕公司 PDF 文档构建智能问答系统。

重点文档：

```text
E:\合兴\吉野家5S管理手册.pdf
```

该 PDF 用于后续开发一个面向 5S 管理手册的智能问答 demo。目标不是单纯“能回答”，而是要做到：

- 能正确解析 PDF 内容。
- 能合理分块并保留页码、章节等 metadata。
- 能通过检索召回正确上下文。
- 回答时能够引用来源页码或章节。
- 没有依据时能够拒答，而不是强行编造。

## 当前进度

已经完成或讨论过的内容：

- 学习了 RAG 基础流程：数据加载、文本分块、索引构建、检索、生成。
- 学习了 PDF 解析工具和策略，包括 Unstructured、PyMuPDF4LLM、pdfplumber、MinerU、PaddleOCR 等。
- 评估过 `吉野家5S管理手册.pdf` 的文档特征：有文本层、包含图片和表格，第 22 页检查表需要特殊处理。
- 学习了文本分块策略，理解了 `chunk_size`、`chunk_overlap` 和结构化分块的影响。
- 学习了 Embedding 模型选型，包括 `BAAI/bge-small-zh-v1.5`、`bge-m3`、Qwen Embedding 等。
- 学习和实践了 FAISS、Milvus、ES9 等检索和向量数据库相关内容。
- 本地部署并验证过 Milvus Standalone，包括 `milvus-standalone`、`milvus-minio`、`milvus-etcd`。
- 本地部署并验证过 Elasticsearch 9，用于后续 ES9 demo。
- 学习了 Milvus 的 Collection、Schema、Entity、Partition、Alias、索引等概念，并和 MySQL、ES 做过类比。
- 学习了上下文扩展、Sentence Window Retrieval、结构化索引、递归检索。
- 学习了混合检索，理解了 BM25 稀疏检索、Embedding 密集检索、RRF 和加权线性融合。
- 运行并分析过 C3 多模态 Milvus 示例和 C4 混合检索相关内容。
- 修复过 LlamaIndex 版本冲突问题，使 `code/C3/06_recursive_retrieval.py` 可以正常运行。

主要学习笔记位置：

```text
docs/learning-notes/rag-mvp-four-stages.md
```

重要讨论、阶段性理解和工程判断，优先更新到这份文档。

## 技术路线

当前建议路线：

```text
短期：
使用 Python 跑通 RAG 核心链路。

中期：
封装 Python FastAPI RAG 服务。

后期：
通过 Java / Spring Boot 调用 Python RAG 服务，接入企业业务系统。
```

当前 demo 优先考虑：

- 文档解析：PyMuPDF4LLM / PyMuPDFLoader，必要时结合 pdfplumber 处理表格。
- 文本分块：章节优先，长度兜底，表格单独结构化处理。
- Embedding：学习阶段可用 `BAAI/bge-small-zh-v1.5`，后续可评估 `bge-m3` 或 Qwen Embedding。
- 检索：ES9 混合检索优先，BM25 + dense vector + metadata filter。
- 向量数据库：Milvus 用于学习和后续增强，尤其是多向量、多模态和大规模场景。
- 生成：LLM 必须基于上下文回答，无依据时明确说明未找到相关信息。

## 工作习惯

协作时请遵守：

- 在改代码前，先看当前文件和上下文。
- 用户明确说“不要动代码”时，只分析和解释，不修改文件。
- 用户说“开始修改”或“直接做”时，再执行代码或文档变更。
- 重要学习讨论要主动询问或直接归纳进学习笔记。
- 修改学习笔记后，优先提交到 Git。
- 默认在 `dev` 分支工作，并推送到用户 fork 的远程 `dev` 分支。
- 不要随便重置、回滚、删除用户已有改动。
- 运行产物、模型缓存、临时索引目录不要主动提交。
- 遇到依赖或环境问题时，先定位根因，再修复，不要盲目升级包。

## Git 约定

当前用户 fork：

```text
https://github.com/wuning159/all-in-rag.git
```

远程名通常为：

```text
myfork
```

当前推荐推送分支：

```text
dev -> myfork/dev
```

不要默认推到官方仓库 `origin`。

## 本地环境提示

常用 Python 环境：

```text
D:\Miniconda\envs\all-in-rag\python.exe
```

项目目录：

```text
E:\PycharmProjects\all-in-rag
```

Docker 中曾经使用过：

- Elasticsearch 9
- Milvus Standalone
- MinIO
- etcd

启动或停止 Docker 容器前，先确认用户当前是否需要这些服务。不要随意删除容器、数据卷或业务数据。

## 安全约束

- 不要提交 `.env`、API Key、Token、账号密码等敏感信息。
- 不要提交公司原始 PDF 文件，尤其是 `E:\合兴\吉野家5S管理手册.pdf`。
- 不要把本地绝对路径写入业务代码作为长期配置，除非只是学习笔记中的上下文说明。
- 不要把大模型权重、临时索引、运行结果图片等产物主动加入 Git。

## 当前重点

当前阶段重点不是追求复杂 Agent 平台，而是先把 RAG 主链路做稳：

```text
PDF 解析
-> 文本分块
-> Embedding
-> ES9 / Milvus 检索
-> 混合检索与阈值判断
-> LLM 基于上下文回答
-> 引用来源与拒答机制
```

后续如果需要，可以再参考 DeerFlow、LangGraph、Dify、RAGFlow 等更上层的 Agent 或平台能力。
