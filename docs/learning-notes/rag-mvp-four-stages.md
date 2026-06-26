# RAG MVP 四阶段学习笔记

这份笔记用于沉淀学习过程中遇到的关键问题、实验现象和工具选型判断。主线按照 RAG 最小可行系统的四个阶段组织：

1. 数据准备与清洗
2. 索引构建
3. 检索策略优化
4. 生成与提示工程

后续如果某段讨论值得总结，可以直接追加到对应阶段，形成一份可持续更新的工程笔记。

维护约定：后续学习过程中，如果某次讨论形成了重要判断、项目决策、工具选型或可复用经验，就直接更新到本文档；阶段性重要更新同步提交并推送到 GitHub，方便在不同电脑继续学习和开发。

## 1. 数据准备与清洗

数据准备与清洗是 RAG 系统的地基。这个阶段的目标不是简单把文件读出来，而是把原始文档转换成适合检索和生成的结构化或半结构化内容。

常见任务包括：

- 解析 PDF、Word、HTML、Markdown、TXT 等多种格式。
- 保留标题、段落、列表、表格、图片说明、页码、来源等元信息。
- 根据文档结构进行合理切分，而不是只按固定字符数切分。
- 针对扫描件或图片型 PDF 使用 OCR。
- 针对复杂表格、公式、图片混排文档选择更专业的解析工具。

### 1.1 PDF 解析为什么困难

PDF 相比 Markdown、HTML、Word 更难处理，因为它本质上更接近“打印后的页面”。很多 PDF 只保存了文字坐标、字体、图片和绘制信息，并不天然包含清晰的语义结构。

因此复杂 PDF 解析通常不只是“抽文字”，还涉及：

- OCR 文字识别
- 页面布局分析
- 标题、段落、列表识别
- 表格结构还原
- 图片区域识别
- 公式识别
- 阅读顺序恢复
- 跨页段落、跨页表格合并

所以在真实项目中，PDF 处理质量往往决定了后续 chunk、embedding、检索和回答质量的上限。

### 1.2 Unstructured 的 `partition` 与 `partition_pdf`

`unstructured.partition.auto.partition()` 是一个通用入口。对于 PDF 文件，它会自动识别文件类型，然后内部调用 `partition_pdf()`。

默认情况下：

```python
partition(filename=pdf_path, content_type="application/pdf")
```

大致等价于：

```python
partition_pdf(filename=pdf_path, strategy="auto")
```

这里的默认策略是 `auto`，不是固定的 `hi_res`，也不是固定的 `ocr_only`。

在当前实验使用的版本中，PDF 的 `auto` 策略大致逻辑是：

```text
如果需要表格结构推断或图片抽取 -> hi_res
否则如果 PDF 文字层可直接提取 -> fast
否则 -> ocr_only
```

也就是说，对于文字层可提取的 PDF，默认 `partition()` 很可能会走 `fast`，它更快，也更依赖 PDF 自带文本层，而不是 OCR。

### 1.3 `fast`、`hi_res`、`ocr_only` 的区别

`fast`：

直接从 PDF 文本层抽取内容，不做 OCR，也不做复杂版面检测。速度快、依赖少，适合文字型 PDF。但对扫描件、复杂版式、图片和表格结构支持较弱。

`hi_res`：

使用高分辨率版面分析模型识别页面结构，能识别更多元素类型，例如 `Image`、`Table`、`Header`、`Title`、`NarrativeText`、`ListItem` 等。它更适合图文表混排的复杂 PDF，但速度更慢，依赖更多，也更容易受 OCR、Poppler、Tesseract、模型环境影响。

`ocr_only`：

把 PDF 页面当作图片，通过 OCR 识别文字，再做简单元素分类。它适合扫描件或没有文本层的 PDF，但结构信息较弱，表格容易变成散乱文本，图片区域本身也不会被很好理解。

实验中观察到：

- `hi_res` 输出元素更多，类型更丰富，能识别图片、表格、标题、正文等结构。
- `ocr_only` 输出元素更少，主要是 `UncategorizedText`、`Title`、`NarrativeText`、`ListItem`。
- 两者字符总数可能接近，但结构质量和元素粒度差别很大。

这说明“元素数量多”不一定代表“文本质量更好”，但通常代表它捕获了更多版面结构。

### 1.4 为什么 OCR 输出像英文乱码

实验中出现了类似提示：

```text
Warning: No languages specified, defaulting to English.
```

这表示没有显式指定 OCR 语言时，Unstructured 默认使用英文：

```python
languages=["eng"]
```

如果文档包含中文，Tesseract 会用英文模型硬识别中文字符，就容易得到类似英文、符号混杂的乱码。

中文 PDF 建议显式指定语言：

```python
elements = partition_pdf(
    filename=pdf_path,
    strategy="ocr_only",
    languages=["chi_sim", "eng"],
)
```

`hi_res` 如果过程中也需要 OCR，也建议同样指定：

```python
elements = partition_pdf(
    filename=pdf_path,
    strategy="hi_res",
    languages=["chi_sim", "eng"],
)
```

同时需要保证本机已安装 Tesseract 中文语言包，并正确配置 `TESSDATA_PREFIX`。

### 1.5 复杂 PDF 工具选型

对于普通文本型 PDF，`Unstructured` 可以快速完成解析，并且很适合接入 LangChain、LlamaIndex 等 RAG 框架。

但对于复杂 PDF，例如科研论文、技术手册、财报、合同、教材、图文表混排文档，仅依赖 `Unstructured` 可能不够稳定。真实应用中更常见的选择是使用专门的文档解析工具或模型。

常见工具定位：

| 工具 | 定位 | 更适合的场景 |
| --- | --- | --- |
| TextLoader | 基础文本加载 | TXT、简单 Markdown |
| DirectoryLoader | 批量目录加载 | 多文件文档库 |
| Unstructured | 多格式通用解析 | PDF、Word、HTML 的快速接入 |
| PyMuPDF4LLM | PDF 转 Markdown | 技术文档、手册、可提取文本 PDF |
| Marker | PDF 转 Markdown | 论文、书籍、结构化 PDF |
| MinerU | 多模态复杂文档解析 | 学术文献、财报、图文表混排 PDF |
| PaddleOCR / PP-Structure | OCR 与版面/表格结构识别 | 中文 OCR、扫描件、表格文档 |
| LlamaParse | 商业 PDF 深度解析 API | 法律合同、学术论文、需要省心解析的场景 |
| Docling | 企业级文档解析 | 企业合同、报告、IBM 生态 |
| FireCrawlLoader | 网页内容抓取 | 在线文档、新闻、网页知识源 |

一个更稳的复杂 PDF RAG 流水线通常是：

```text
PDF
 -> MinerU / PaddleOCR / Marker / PyMuPDF4LLM 做结构化解析
 -> Markdown / JSON / HTML
 -> 按标题、段落、表格、图片说明进行结构化 chunk
 -> embedding
 -> 向量库 / 混合索引
 -> 检索与重排
 -> LLM 生成答案
```

相比直接：

```text
PDF -> Unstructured -> chunk -> embedding
```

这种方式前处理更重，但对复杂文档的质量上限更高。

### 1.6 当前阶段的实践判断

如果只是学习或快速验证 RAG 流程：

- 使用 `Unstructured` 足够。
- 使用 `partition()` 或 `partition_pdf(strategy="auto")` 可以快速开始。
- 对英文或文本型 PDF，优先尝试默认 `auto` 或 `fast`。

如果处理中文扫描件：

- 使用 `ocr_only`。
- 显式设置 `languages=["chi_sim", "eng"]`。
- 确认 Tesseract、Poppler、语言包和环境变量都配置正确。

如果处理复杂图文表混排 PDF：

- 优先尝试 `hi_res`。
- 需要表格时加 `infer_table_structure=True`。
- 对重要业务场景，考虑 MinerU、PaddleOCR、Marker、PyMuPDF4LLM、LlamaParse、Docling 等专门工具。

核心结论：

```text
PDF 解析不是单纯的文件加载问题，而是文档理解问题。
复杂 PDF 的 RAG 效果，往往先取决于解析质量，再取决于检索和生成。
```

## 2. 索引构建

索引构建阶段的核心任务，是把清洗和切分后的文本转换成可检索的结构。最常见的做法是：

```text
文本 chunk -> embedding 模型 -> 向量 -> 向量数据库 / 混合索引
```

这里的 embedding 模型，就是把自然语言文本转换成向量的模型。后续向量检索能不能召回正确内容，很大程度上取决于：

- chunk 切得是否合理
- embedding 模型是否适合当前语言和业务领域
- 向量库索引和相似度计算是否配置正确
- 是否保留了来源、页码、标题层级等 metadata

### 2.1 为什么文本分块和 embedding 模型有关

文本切分的两个首要原因是：

1. embedding 模型通常有最大输入长度限制。
2. LLM 生成答案时也有上下文窗口限制。

如果 chunk 太大，可能超过 embedding 模型的 token 限制，或者虽然没有超过限制，但一个向量里混入太多主题，导致语义被稀释，检索时不够精准。

如果 chunk 太小，又可能把一个完整语义拆碎，导致召回的片段缺少上下文，LLM 虽然拿到了相关词，但拿不到完整答案。

所以 chunk_size 和 chunk_overlap 的设置，本质上是在平衡：

- 单个 chunk 的语义完整性
- embedding 向量的表达质量
- 检索召回的精度
- LLM 最终可用上下文的长度

### 2.2 企业级 embedding 模型选型

企业级 embedding 模型大致可以分成两类：

| 类型 | 代表模型 / 服务 | 适合场景 |
| --- | --- | --- |
| 商业 API | OpenAI text-embedding-3-small / text-embedding-3-large | 通用英文、多语言、SaaS 产品、快速上线 |
| 商业 API | Cohere Embed v4 | 多语言、长上下文、文本和图像混合检索 |
| 商业 API | Voyage AI embedding 系列 | 法律、金融、代码等垂直领域检索 |
| 云厂商 API | Google Vertex AI Gemini Embedding | Google Cloud 体系内应用、多语言和代码场景 |
| 云厂商 API | Amazon Titan Text Embeddings V2 | AWS / Bedrock 体系内应用 |
| 可私有化部署 | BAAI bge-m3 | 中文、多语言、dense/sparse/ColBERT 多模式检索 |
| 可私有化部署 | Jina Embeddings v3 | 多语言、长文本、支持任务类型参数 |
| 可私有化部署 | Qwen3 Embedding | 中文、多语言、代码、阿里生态和私有部署场景 |

学习阶段可以先用轻量模型，例如：

```text
BAAI/bge-small-zh-v1.5
```

它足够轻，适合本地跑通 RAG 全流程。

如果进入更接近真实业务的中文知识库，可以优先考虑：

```text
bge-m3
Qwen3 Embedding
Jina Embeddings v3
```

如果数据安全要求高，或者文档不能出内网，更适合选择可私有化部署的 embedding 模型。如果追求省心、稳定和快速上线，商业 API 会更方便。

### 2.3 选 embedding 模型时不要只看榜单

embedding 模型的公开榜单只能作为参考，不能直接等同于业务效果。RAG 里真正重要的是它在自己数据集上的召回质量。

更可靠的评估方式是自己准备一小批测试问题，例如 30 到 100 个问题，并标注每个问题应该命中的文档片段，然后比较不同模型的：

- Recall@K：前 K 个召回结果里是否包含正确答案片段
- MRR：正确答案是否排得足够靠前
- 中文、英文、代码、表格、术语等不同内容类型的表现
- 向量维度、速度、成本、部署复杂度

一个实用判断是：

```text
如果模型召回不到正确 chunk，后面的 LLM 再强也只能基于错误上下文回答。
```

所以在 RAG 系统里，embedding 模型不是一个可随便替换的小组件，而是索引质量和检索质量的基础。

### 2.4 当前主流 embedding 模型对比

`all-in-rag` 第 3 章的“嵌入模型选型指南”强调，模型选择不能只看一个总榜分数，而要综合看：

- Retrieval 检索任务表现
- 是否支持中文 / 多语言
- 模型大小和部署成本
- 向量维度
- 最大 token 长度
- API 成本或本地推理成本
- 是否适合自己的业务数据

当前市面上较常见、较值得关注的 embedding 模型可以分成两类：商业 API 和可私有化部署模型。

| 模型 / 服务 | 类型 | 典型维度 / 长度 | 优势 | 局限 | 适合场景 |
| --- | --- | --- | --- | --- | --- |
| OpenAI `text-embedding-3-small` | API | 1536 维，最大输入 8192 tokens | 成本低、稳定、通用效果好 | 数据需要走外部 API | 通用 RAG、英文/多语言知识库、快速上线 |
| OpenAI `text-embedding-3-large` | API | 3072 维，最大输入 8192 tokens，可降维 | 质量更高，可用 `dimensions` 控制维度 | 成本高于 small | 对召回质量要求更高的 SaaS / 企业知识库 |
| Cohere `embed-v4.0` | API | 256/512/1024/1536 维，最大上下文 128k | 支持文本、图片、混合输入，长上下文强 | 商业 API，成本和数据出域要考虑 | 多语言、长文档、多模态检索 |
| Voyage `voyage-4` / `voyage-4-large` | API | 默认 1024 维，可选 256/512/2048，最大 32k tokens | 检索质量强，有 query/document 输入类型 | 商业 API | 高质量通用检索、法律/金融/代码等垂直检索 |
| Google `gemini-embedding-001` | API | 最高 3072 维，最大 2048 tokens | Google Cloud 体系内集成方便，英文/多语言/代码能力强 | 云平台绑定较强 | Google Cloud 上的 RAG 应用 |
| Amazon Titan Text Embeddings V2 | API | 默认 1024 维，可选 512/256，最大 8192 tokens | AWS Bedrock 集成方便 | 主要优化英文，多语言要实际测试 | AWS 体系内知识库 |
| BAAI `bge-m3` | 开源 / 私有化 | 1024 维，最大 8192 tokens | 中文和多语言友好，支持 dense/sparse/ColBERT 多模式 | 本地部署需要资源和调优 | 中文企业知识库、混合检索、私有化 RAG |
| Qwen3 Embedding | 开源 / 私有化 | 0.6B: 1024 维；4B: 2560 维；8B: 4096 维，最大 32k | 中文、多语言、代码检索强，支持 instruction，配套 reranker | 4B/8B 资源要求较高 | 中文企业知识库、代码/多语言检索、私有部署 |
| Jina Embeddings v5 text / v5 omni | API / 可部署 | v5 text 支持 32k 上下文；v5 omni 支持文本/图像/音频/视频统一空间 | 多语言、长上下文、多模态能力强 | 新模型需要结合业务测试稳定性 | 长文档、多模态、跨语言 RAG |
| multilingual-e5 系列 | 开源 / 私有化 | 常见 384/1024 维，长度通常较短 | 经典、轻量、生态成熟 | 长文本能力弱于新一代模型 | 本地学习、小型多语言检索 |

对当前 `吉野家5S管理手册` demo，可以按阶段选择：

```text
学习 / 快速跑通：
BAAI/bge-small-zh-v1.5

中文手册问答质量优先：
bge-m3 或 Qwen3-Embedding-0.6B

如果后续要做更强召回：
Qwen3-Embedding-4B/8B + Qwen3-Reranker

如果公司允许外部 API：
OpenAI text-embedding-3-small / large、Cohere embed-v4.0、Voyage voyage-4
```

结合当前周三 demo 和 ES9，建议先选一个维度稳定、部署不复杂的中文模型，例如：

```text
BAAI/bge-small-zh-v1.5
```

或者如果机器资源允许：

```text
bge-m3
```

原因是 ES9 的 `dense_vector` mapping 需要提前确定向量维度。embedding 模型一旦换了，向量维度和语义空间都可能改变，通常需要：

```text
删除旧索引 / 新建索引
重新 embedding 所有 chunk
重新写入 ES9
```

因此 demo 阶段不要频繁换模型。更稳的方式是先用一个模型把全流程跑通，再用同一批问题做横向对比。

推荐评测方式：

```text
同一份 PDF
同一套 chunk
同一组测试问题
分别使用不同 embedding 模型入库
比较 Recall@K、MRR、回答引用页码准确率、延迟和部署成本
```

对于这个项目，第一批测试问题可以包括：

- “5S 是什么意思？”
- “5S 实施步骤有哪些？”
- “餐厅总经理负责什么？”
- “物品有名有家有哪些要求？”
- “整理 10 分包含哪些检查项？”
- “清扫部分怎么扣分？”

当前判断：

```text
不要只问“哪个 embedding 最强”。
应该问“在我的中文手册、我的 chunk、我的 ES9 检索方式、我的测试问题上，哪个模型召回最稳”。
```

## 3. 检索策略优化

待补充。

## 4. 生成与提示工程

待补充。

## 5. 项目实践：吉野家5S管理手册智能问答

这个章节用于沉淀后续围绕 `吉野家5S管理手册.pdf` 开发智能问答系统时的判断、实验记录和工程决策。

### 5.1 PDF 文档特征评估

目标文件：

```text
E:\合兴\吉野家5S管理手册.pdf
```

已观察到的文档特征：

- 共 22 页。
- 文件不是纯扫描件，存在可提取文本层。
- 文本层可提取约 9,194 个字符。
- 文档类型是中文管理手册，包含目录、章节标题、正文说明、项目符号、组织结构图、示例图片和检查表。
- 前半部分主要是 5S 概念、制度、流程、标准说明。
- 中后段包含较多图片示例。
- 第 22 页是复杂检查表，表格结构比较重要。

这说明该 PDF 不适合直接按“扫描件”处理，也不应该默认先走 OCR。它更像是：

```text
可提取文本层的中文手册 + 图片示例 + 局部复杂表格
```

### 5.2 文档加载器选择

当前推荐优先级：

| 目标 | 推荐方案 | 原因 |
| --- | --- | --- |
| 构建普通 RAG 问答 | PyMuPDF4LLM / PyMuPDFLoader | 文档有文本层，适合稳定提取正文、标题和页码 |
| 跟随课程快速实验 | Unstructured `partition_pdf(strategy="fast")` | 能快速利用 PDF 文本层，依赖较少，速度快 |
| 识别标题、图片、表格区域等元素类型 | Unstructured `partition_pdf(strategy="hi_res")` | 能获取更丰富的页面结构，但速度慢、依赖多 |
| 精确保留最后检查表的行列结构 | pdfplumber / Camelot / MinerU / Docling | 第 22 页表格结构复杂，普通文本抽取可能丢失行列关系 |
| 处理纯扫描件或无文本层 PDF | PaddleOCR / `ocr_only` | 当前 PDF 不是这种情况，因此 OCR 不是首选 |

当前判断：

```text
不建议优先使用 ocr_only。
```

原因是该 PDF 已经存在文本层。OCR 会把清晰文本重新当图片识别一遍，速度更慢，还可能造成中文、符号、表格顺序识别错误。OCR 更适合作为无文本层扫描件的兜底方案。

更适合当前文档的基础路线是：

```text
吉野家5S管理手册.pdf
 -> PyMuPDF4LLM 转 Markdown
 -> 按标题和章节切分
 -> 保留 source、page、section 等 metadata
 -> embedding
 -> 向量库 / 混合索引
 -> 检索
 -> LLM 基于上下文回答
```

### 5.3 对最后一页检查表的特殊处理

第 22 页是检查表，包含分值、分类、检查项和问题说明等结构。如果后续问题只是：

- “整理 10 分有哪些要求？”
- “清扫 30 分包含哪些检查项？”
- “5S 检查评估表包含哪些维度？”

普通文本抽取大概率可以满足基本问答。

但如果后续需要更精确地回答：

- 某个检查项属于哪个一级分类
- 每一项对应几分
- 某个扣分项属于哪一行
- 把检查表恢复成结构化表格

那么第 22 页最好单独做结构化解析，输出成 JSON、CSV 或 Markdown 表格，再进入 RAG。

更稳的处理方式是：

```text
第 1-21 页：按普通手册文档处理
第 22 页：单独表格解析，保留行列关系
```

对于当前这个 PDF，第 22 页更推荐优先尝试 `pdfplumber`，而不是一上来使用 OCR 或大型文档解析工具。

原因是：

- 该 PDF 有可提取文本层。
- 第 22 页的表格线比较清晰。
- `pdfplumber` 对“文本层 + 明确表格线”的 PDF 表格比较合适。
- 它轻量、可控，方便把表格转成 JSON、CSV 或 Markdown 表格。
- OCR 会重新识别已经存在的清晰文本，可能引入中文识别错误和阅读顺序错误。

推荐优先级：

| 优先级 | 工具 | 判断 |
| --- | --- | --- |
| 1 | pdfplumber | 当前首选，适合有文本层、有清晰表格线的 PDF 表格 |
| 2 | Camelot | 也适合有线表格，但安装依赖和调参成本略高 |
| 3 | MinerU / Docling / Marker | 如果轻量表格工具抽取效果不好，再升级到文档解析工具 |
| 4 | PaddleOCR / OCR | 只在表格是扫描图、没有文本层时优先考虑 |

一个适合 RAG 的结构化结果示例：

```json
{
  "一级分类": "整理",
  "分值": "10分",
  "检查项": "物品清理",
  "标准": "各功能间内无多余物品",
  "扣分规则": "发现1处扣1分",
  "来源页码": 22
}
```

这类结构化数据比直接把整张表当作一段普通文本更适合问答。比如用户问“整理 10 分有哪些检查项”“清扫部分怎么扣分”“物品标示标签属于哪一类”，系统更容易检索到准确行。

需要注意的是，`all-in-rag` 教程文档里已经讲了 PDF 加载器的大方向，例如 `PyMuPDF4LLM`、`Unstructured`、`MinerU`、`Docling`、`PaddleOCR` 等，也提到 `partition_pdf` 支持表格结构推理。但教程没有展开到 `pdfplumber` / `Camelot` 这种“单页表格抽取工具”的细分选择。

所以这里的判断属于项目实践补充：

```text
教程层面：讲 PDF 文档加载器和复杂 PDF 工具选型。
当前项目：第 22 页是有文本层的清晰检查表。
工程选择：先试 pdfplumber，失败再升级 MinerU / Docling / Marker。
```

### 5.4 后续智能问答系统的初步开发路线

这个 PDF 对应的智能问答系统，可以先做一个最小可行版本：

1. 文档解析：优先用 PyMuPDF4LLM 将 PDF 转成 Markdown。
2. 文本清洗：去掉重复页眉、页脚、水印、版权说明等噪声。
3. 文本切分：按标题层级优先切分，再用 chunk_size 和 chunk_overlap 控制长度。
4. 元数据保留：每个 chunk 保留 PDF 文件名、页码、章节标题。
5. 向量化：先用轻量中文 embedding 模型跑通流程，例如 `BAAI/bge-small-zh-v1.5`。
6. 检索：先做向量检索，后续再考虑关键词 + 向量的混合检索。
7. 生成：Prompt 要求模型只能基于手册内容回答，不知道就说明没有在手册中找到。
8. 评估：准备一批围绕 5S 定义、实施步骤、检查表、岗位职责的测试问题。

初版目标不是追求复杂，而是先验证：

```text
用户问一个关于 5S 管理手册的问题，系统能召回正确页码和章节，并基于原文给出可靠回答。
```

如果初版效果稳定，再逐步增强：

- 加入检查表结构化解析。
- 加入引用页码。
- 加入重排序模型。
- 加入混合检索。
- 加入前端问答界面。
- 加入问题日志和失败案例分析。

### 5.5 针对本 PDF 的分块策略

这个 PDF 不建议只使用固定长度分块，例如直接每 800 字切一段。更合适的策略是：

```text
章节优先 + 长度兜底 + 表格单独处理
```

也就是：

```text
第 1-21 页：按标题和章节结构切分为主
第 22 页：表格结构化后，按检查项或分类切分
```

原因是 `吉野家5S管理手册.pdf` 是一份管理手册，本身有比较清晰的章节结构，例如：

- “5S”概述
- “5S”制度建立
- “5S”实施步骤
- “5S”标准
- 追踪与检查

如果直接按固定长度切分，可能会把一个完整小节从中间切断。这样用户问“5S 实施步骤有哪些”时，系统可能只召回到其中一半内容，导致回答不完整。

更合理的做法是先尊重文档本身的语义结构：

```text
先按标题、章节、列表、表格等结构切分
再用 chunk_size 控制最大长度
```

初版推荐流程：

```text
PDF
 -> PyMuPDF4LLM 转 Markdown
 -> MarkdownHeaderTextSplitter 按标题层级切分
 -> RecursiveCharacterTextSplitter 做长度兜底
 -> 第 22 页检查表单独结构化
 -> 每个 chunk 保留 source、page、section metadata
```

普通正文的初始参数可以先用：

```python
chunk_size = 800
chunk_overlap = 100
```

这里的 `800/100` 不是绝对标准，而是一个适合中文手册问答的初始值：

- `chunk_size=800`：通常足够容纳一个小节或一组完整说明，不至于太碎。
- `chunk_overlap=100`：可以保留相邻片段之间的上下文，减少边界处信息丢失。
- 如果某个章节本身很短，就不要为了凑长度和后面章节强行合并。
- 如果某个章节超过 800 到 1000 字，再用递归切分器继续切。

具体规则可以这样理解：

| 内容类型 | 分块策略 | 原因 |
| --- | --- | --- |
| 目录 | 可保留为结构参考，也可以不入库 | 目录对问答价值有限，但有助于理解章节结构 |
| 普通正文 | 按标题层级切分，再做长度控制 | 保留语义完整性 |
| 项目符号 / 推行要领 | 尽量保持同一组列表完整 | 列表被拆散后容易影响回答完整性 |
| 图片附近说明 | 保留图片附近的文字说明 | 当前初版不做图像理解，先利用文字说明 |
| 第 22 页检查表 | 表格结构化后按一级分类、检查项切分 | 表格行列关系比普通段落更重要 |

第 22 页检查表不应该当普通长文本切分。更推荐先解析成结构化数据，例如：

```json
{
  "一级分类": "整理",
  "分值": "10分",
  "检查项": "物品清理",
  "标准": "各功能间内无多余物品",
  "扣分规则": "发现1处扣1分",
  "来源页码": 22
}
```

然后再按行或按二级分类形成 chunk。这样用户问：

- “整理 10 分有哪些检查项？”
- “清扫部分怎么扣分？”
- “物品标示标签属于哪一类？”

系统更容易召回到准确的检查项，而不是召回一整页混在一起的表格文本。

最终判断：

```text
固定长度分块适合作为兜底，不适合作为这个 PDF 的第一分块策略。
这个项目应优先使用结构化分块：章节按标题切，检查表按行列结构切。
```

### 5.6 向量数据库选型与 ES9 Demo 决策

RAG 的检索环节通常以 embedding 语义搜索为核心，可以拆成两条链路：

```text
离线索引构建：
文档 -> 加载 -> 清洗 -> 分块 -> embedding -> 向量库 / 检索引擎

在线查询检索：
用户问题 -> 同一个 embedding 模型 -> 问题向量 -> 相似度检索 -> Top-K chunk -> LLM 生成答案
```

这里有一个重要原则：

```text
文档 chunk 用什么 embedding 模型入库，用户问题通常也要用同一个 embedding 模型向量化。
```

原因是不同 embedding 模型会形成不同的向量空间。如果文档向量用 `bge-small-zh-v1.5` 生成，而查询向量用另一个不兼容的模型生成，相似度计算就不再可靠。

如果后续更换 embedding 模型，通常需要重新执行：

```text
所有 chunk 重新 embedding -> 重新写入索引
```

概念上，向量检索是在计算“问题向量”和“文档块向量”的相似度。工程实现上，向量数据库通常不会真的逐个暴力比较所有向量，而是使用 ANN（Approximate Nearest Neighbor，近似最近邻）索引来加速检索，例如 HNSW、IVF、DiskANN 等。

常见向量数据库和检索方案对比：

| 工具 | 定位 | 优势 | 局限 | 当前项目建议 |
| --- | --- | --- | --- | --- |
| FAISS | 单机向量检索库 | 快、经典、适合本地实验和算法验证 | 不是完整数据库，metadata、权限、服务化能力弱 | 学习和本地验证可以用 |
| Chroma | 轻量 RAG 向量库 | 上手快，适合 demo、小型知识库、LangChain 生态友好 | 大规模生产能力相对弱 | 学习阶段可用 |
| Qdrant | 开源向量数据库 | 部署简单，payload/filter 能力好，适合中小型生产 RAG | 超大规模生态厚度不如 Milvus | 后续替代 ES 做纯向量方案时可考虑 |
| Milvus | 企业级分布式向量数据库 | 大规模、高性能、索引类型多，生态成熟 | 部署和运维复杂度更高 | 大规模知识库再考虑 |
| Weaviate | AI 原生向量数据库 | schema、语义搜索、混合搜索和 RAG 能力完整 | 概念和配置较多 | 适合做完整 AI 应用后端 |
| Pinecone | 托管向量数据库 | 省运维，生产化能力强 | 商业服务，成本和数据出域要考虑 | 外部 SaaS 场景可考虑 |
| pgvector | PostgreSQL 向量扩展 | 如果已有 PostgreSQL，架构简单 | 超大规模向量检索性能不如专用向量库 | 业务数据强依赖 PG 时可考虑 |
| Elasticsearch 9 / ES9 | 搜索引擎 + 向量检索 | 关键词检索、过滤、聚合、权限、运维生态成熟，同时支持 dense vector/kNN | 纯向量能力不是最轻量，mapping 和检索参数需要设计 | 当前 demo 优先选 |

当前领导侧已经倾向使用 ES9，并且本地已经安装好 ES9。因此周三 demo 的策略应该是：

```text
不要为了“向量库最优”而引入新组件。
先基于 ES9 做出可演示、可解释、可扩展的 RAG Demo。
```

选择 ES9 做 demo 的理由：

- 领导已经明确关注 ES9，演示更贴近预期。
- ES 本身擅长关键词检索、字段过滤、分页、聚合和工程化管理。
- ES9 支持 dense vector / kNN，可承载基础向量检索。
- 对中文手册问答来说，关键词检索和向量检索可以互补。
- 后续可以自然扩展为混合检索：BM25 + 向量相似度 + rerank。
- 运维和团队接受度通常比新引入 Milvus/Qdrant 更高。

当前 `吉野家5S管理手册` 的 demo 推荐路线：

```text
PDF
 -> PyMuPDF4LLM 解析为 Markdown
 -> 第 22 页 pdfplumber 单独结构化
 -> 章节优先分块
 -> embedding 生成向量
 -> 写入 ES9 index
      - content: chunk 文本
      - content_vector: dense_vector
      - source: PDF 文件名
      - page: 页码
      - section: 章节
      - chunk_type: normal_text / table_row
 -> 用户问题 embedding
 -> ES9 kNN / hybrid search 召回 Top-K
 -> 把召回内容和原始问题交给 LLM
 -> 返回答案 + 来源页码
```

ES9 索引设计需要特别注意：

| 字段 | 作用 |
| --- | --- |
| `content` | 原始 chunk 文本，用于展示、BM25、LLM 上下文 |
| `content_vector` | embedding 向量，用于 kNN 检索 |
| `source` | 文档来源，例如 `吉野家5S管理手册.pdf` |
| `page` | 页码，用于答案引用 |
| `section` | 章节标题，用于过滤和展示 |
| `chunk_type` | 区分普通正文、表格行、图片说明等 |
| `table_category` | 第 22 页检查表的一级分类，可选 |
| `score_rule` | 检查表扣分规则，可选 |

对于 demo，检索策略可以先做三档：

1. 纯向量检索：验证语义搜索是否能召回正确段落。
2. 关键词检索：验证用户输入手册原词时 ES 的传统搜索能力。
3. 混合检索：把 BM25 和向量检索结果合并，提升稳定性。

演示时可以准备几类问题：

| 问题类型 | 示例 |
| --- | --- |
| 定义类 | “5S 是什么意思？” |
| 流程类 | “5S 实施步骤有哪些？” |
| 职责类 | “餐厅总经理在 5S 中负责什么？” |
| 标准类 | “物品有名有家有哪些要求？” |
| 检查表类 | “整理 10 分包含哪些检查项？” |
| 扣分类 | “清扫部分发现问题怎么扣分？” |

当前阶段的判断：

```text
学习和快速验证：Chroma / FAISS 很轻。
通用生产向量库：Qdrant / Milvus / Weaviate 更专业。
当前周三 demo：优先 ES9，因为它已经安装好、领导关注、且适合展示关键词 + 向量混合检索。
```

后续如果 ES9 demo 跑通，再考虑把同一份 chunk 数据复制到 Qdrant 或 Milvus，做横向对比实验：

```text
同一批 chunk
同一个 embedding 模型
同一组测试问题
比较 ES9 / Qdrant / Milvus 的召回质量、延迟、部署复杂度和维护成本
```

这样对比才公平，因为变量只剩下“检索后端”，不会把解析、分块、embedding 模型差异混在一起。

参考资料：

- Elastic 官方文档：dense vector search、kNN、hybrid search。
- Chroma 官方文档：embedding 存储、metadata filtering、dense/sparse/hybrid search。
- Qdrant 官方文档：向量数据库、payload/filter、相似度检索。
- Milvus 官方文档：企业级向量数据库、分布式索引和检索。
- Weaviate 官方文档：语义搜索、混合搜索、RAG 后端。
- Pinecone 官方文档：托管向量数据库、metadata filtering、rerank、hybrid search。
- pgvector / FAISS 官方项目：PostgreSQL 向量扩展和本地向量检索库。

### 5.7 误召回、不知道机制与 Demo 检索设计

在 `code/C3/02_langchain_faiss.py` 的 FAISS 示例中，知识库只有三条文本：

```python
texts = [
    "张三是法外狂徒",
    "FAISS是一个用于高效相似性搜索和密集向量聚类的库。",
    "LangChain是一个用于开发由语言模型驱动的应用程序的框架。"
]
```

如果查询：

```text
FAISS 是做什么的？
```

可以正确召回：

```text
FAISS是一个用于高效相似性搜索和密集向量聚类的库。
```

如果查询：

```text
张三是谁？
```

可以召回：

```text
张三是法外狂徒
```

但如果查询：

```text
李四是谁？
```

知识库里其实没有“李四”，FAISS 仍然会返回最相似的一条，很可能还是：

```text
张三是法外狂徒
```

这个现象说明：

```text
向量检索负责找“最相似”，不负责判断“知识库里到底有没有正确答案”。
```

FAISS 默认的 `similarity_search(query, k=1)` 会返回最相似的 1 条。即使所有候选都不是真正答案，它也会返回距离最近的那条。

在 LangChain 的 FAISS 默认实现中，常见距离策略是 L2 欧氏距离：

```text
距离越小，向量越接近。
```

但需要注意：

```text
相似不等于正确。
```

例如“张三是谁？”和“李四是谁？”句式非常接近，embedding 模型会认为它们语义相似，因为都是在问“某个人是谁”。如果知识库里只有张三，没有李四，单纯向量检索就可能误召回“张三”。

这不是 FAISS 算错了，而是最小示例缺少生产级 RAG 必须具备的防护机制。

高可靠 RAG 不能只依赖 LLM 在最后说“不知道”，检索层也要尽量减少错误上下文进入生成阶段。

推荐的防护策略：

| 策略 | 作用 |
| --- | --- |
| Top-K 多召回 | 不只拿 1 条，先拿多条候选，避免过早锁死结果 |
| 相似度阈值 | 如果最相关结果也低于阈值，就认为没有可靠召回 |
| BM25 + 向量混合检索 | 关键词精确匹配和语义匹配互补 |
| 实体一致性检查 | 问“李四”时，召回内容至少要包含“李四”或等价实体 |
| metadata 过滤 | 按文档、章节、页码、表格类型等过滤候选范围 |
| reranker 二次排序 | 用更强模型判断 query 和 chunk 是否真的相关 |
| LLM 上下文约束 | Prompt 要求只能基于召回内容回答，不知道就说明未找到 |

其中，对“李四是谁？”这种问题，最关键的是：

```text
实体一致性检查 + 不知道机制
```

逻辑可以是：

```text
问题中出现实体“李四”
召回 chunk 中没有“李四”
=> 不允许直接把“张三是法外狂徒”作为答案
=> 返回：知识库中没有找到关于“李四”的信息
```

这也是为什么生产级 RAG 往往会使用混合检索：

```text
向量检索：擅长语义相似。
BM25：擅长关键词精确匹配。
混合检索：同时利用语义召回和关键词约束。
```

对于当前 `吉野家5S管理手册` Demo，检索链路建议设计为：

```text
用户问题
 -> query 预处理
      - 提取关键词 / 实体 / 分值 / 章节名
 -> ES9 混合检索
      - BM25 检索 content
      - kNN 检索 content_vector
      - 按 page / section / chunk_type 做可选过滤
 -> Top-K 候选合并
 -> 阈值判断
 -> 实体 / 关键词一致性检查
 -> reranker 可选
 -> 组装上下文
 -> LLM 生成答案
      - 必须基于上下文
      - 没有依据就说“手册中未找到相关信息”
 -> 返回答案 + 引用页码 / 章节
```

Demo 可以先实现三个检索模式，方便演示对比：

| 模式 | 说明 | 演示价值 |
| --- | --- | --- |
| 纯向量检索 | 只用 embedding 相似度召回 | 展示语义搜索能力，也能展示误召回风险 |
| 关键词检索 | 只用 BM25 | 展示精确词匹配能力 |
| 混合检索 | BM25 + 向量检索 | 展示更适合业务知识库的折中方案 |

Demo 中建议专门准备“反例问题”，用来展示不知道机制：

| 问题 | 期望行为 |
| --- | --- |
| “5S 是什么意思？” | 正常回答并引用手册页码 |
| “整理 10 分包含哪些检查项？” | 从第 22 页检查表回答 |
| “李四是谁？” | 返回手册中未找到相关信息 |
| “公司食堂几点开门？” | 返回手册中未找到相关信息 |
| “6S 是什么？” | 如果手册没有 6S，不要强行用 5S 回答 |

这个设计的目的不是让系统“永远有答案”，而是让系统在没有依据时能够可靠拒答。

当前阶段的工程判断：

```text
周三 Demo 不应该只展示“能回答”。
也应该展示“不能回答时不会胡答”。
这比单纯跑通向量检索更接近真实业务要求。
```

后续如果要进一步增强，可以加入：

- reranker：对 Top-K 结果二次排序。
- query rewrite：将用户口语问题改写成更适合检索的问题。
- query classification：先判断问题是定义类、流程类、检查表类还是无关问题。
- answer validation：生成后再检查答案是否有上下文依据。

### 5.8 RAG 框架选型：Python 与 Java

RAG 框架可以分成三类来看：

```text
开发框架：帮助开发者组合 loader、chunk、embedding、retriever、LLM、agent。
RAG 专用框架：更关注文档索引、检索、问答、评估和生产化链路。
低代码 / 平台型产品：更适合快速搭建应用、工作流和管理界面。
```

Python 生态更成熟，适合快速验证和深度定制；Java 生态更适合接入企业已有系统，例如 Spring Boot、权限、流程、后台管理和内部服务治理。

Python 侧常见选择：

| 框架 / 平台 | 定位 | 优势 | 局限 | 适合场景 |
| --- | --- | --- | --- | --- |
| LangChain / LangGraph | LLM 应用和 Agent 编排框架 | 生态大、组件多、工具链丰富，LangGraph 适合复杂流程和状态机 | 抽象较多，版本变化快，工程上需要约束用法 | 需要灵活组合 RAG、工具调用、Agent 工作流 |
| LlamaIndex | 数据框架和 RAG 框架 | 围绕数据接入、索引、retriever、query engine 的抽象更贴近 RAG | 深度定制时也需要理解内部抽象 | 文档问答、知识库、多索引、多数据源 RAG |
| Haystack | 生产化 NLP / RAG Pipeline 框架 | pipeline 思路清晰，适合组件化编排和生产服务 | 国内教程相对少一些 | 生产级问答、搜索增强、可观测 pipeline |
| DSPy | LLM 程序优化框架 | 强调 prompt / pipeline 的自动优化和评估 | 对初学者门槛较高，不是传统 loader-to-vectorstore 框架 | 后期做检索、重排、生成策略优化 |
| RAGFlow | 面向 RAG 的开源平台 | 内置文档解析、知识库、检索、问答界面，更接近产品形态 | 二次开发要适应其平台架构 | 想快速搭一个完整 RAG 知识库产品 |
| Dify | LLM 应用开发平台 | 工作流、应用管理、模型接入、知识库能力完整 | 深度底层检索策略不如自研灵活 | 快速做原型、内部应用、运营管理界面 |

Java 侧常见选择：

| 框架 | 定位 | 优势 | 局限 | 适合场景 |
| --- | --- | --- | --- | --- |
| Spring AI | Spring 体系内的 AI 应用框架 | 和 Spring Boot、配置、Bean、企业工程体系结合自然 | AI 生态丰富度不如 Python | Java 企业项目中接入 LLM、Embedding、Vector Store |
| LangChain4j | Java 版 LLM 应用框架 | 面向 Java 开发者，提供模型、embedding、retriever、tool、memory 等抽象 | 生态规模小于 Python LangChain | Java 服务中直接实现 RAG 或 Agent 能力 |
| Semantic Kernel Java | 微软 Semantic Kernel 的 Java 实现 | 适合函数调用、插件、规划和企业系统集成 | Java 生态成熟度仍需结合版本评估 | 微软技术栈、插件化 Agent、企业编排 |
| Quarkus LangChain4j | Quarkus 对 LangChain4j 的集成 | 适合云原生、轻量 Java 服务 | 更偏 Quarkus 技术栈 | 已经使用 Quarkus 的团队 |

框架选型不要只看“哪个最火”，要看当前系统的主要矛盾：

| 主要目标 | 推荐方向 |
| --- | --- |
| 快速学习 RAG 全链路 | LangChain 或 LlamaIndex |
| 文档知识库问答质量优先 | LlamaIndex / Haystack / 自研 Python pipeline |
| 复杂 Agent 工作流 | LangGraph，后续再参考 DeerFlow 这类 Agent 平台 |
| 快速做可演示产品 | Dify / RAGFlow |
| 企业 Java 系统集成 | Spring AI / LangChain4j |
| 公司已有 Spring Boot 主系统 | Java 调 Python RAG 服务，或者 Spring AI 只做调用层 |

对于当前 `吉野家5S管理手册` Demo，推荐路线是：

```text
短期：Python 自研轻量 RAG pipeline
  - PDF 解析、chunk、embedding、ES9 检索、reranker、LLM 回答都放在 Python 里
  - 先保证准确率、引用页码、拒答机制

中期：封装为 FastAPI 服务
  - Java / Spring Boot 通过 HTTP 调用
  - 返回 answer、citations、retrieved_chunks、answerable

后期：根据企业技术栈选择
  - 如果继续 Python 为主：可引入 LlamaIndex / LangChain / Haystack 做更完整编排
  - 如果 Java 侧要承担更多 AI 能力：评估 Spring AI 或 LangChain4j
  - 如果要快速产品化管理后台：评估 Dify / RAGFlow
```

当前不建议一开始就把框架堆满。周三 Demo 的关键是证明核心链路可靠：

```text
文档解析准确
chunk 可追溯
检索能命中正确页
不知道时能拒答
回答能带引用
```

框架是为了降低工程复杂度，不是为了替代检索质量本身。RAG 的效果上限仍然主要取决于文档解析、分块、embedding、检索策略、重排和提示词约束。

### 5.9 Python RAG 服务与 Java 业务系统集成

当前 RAG Demo 使用 Python 开发是合理的，因为 Python 在 AI 和文档处理生态上更成熟：

- LangChain / LlamaIndex
- PyMuPDF4LLM / pdfplumber / Unstructured / MinerU
- sentence-transformers / transformers
- pymilvus / Elasticsearch Python client
- reranker、embedding、多模态模型生态

但企业内部系统往往更偏 Java 技术栈，例如 Spring Boot、权限系统、用户体系、工作流、后台管理系统等。因此更适合采用分层集成，而不是一开始就把所有 RAG 能力改写成 Java。

当前推荐架构是：

```text
Java 业务系统
  -> 负责用户、权限、菜单、流程、业务接口、前端集成
  -> 通过 HTTP 调用 Python RAG 服务

Python RAG 服务
  -> 负责 PDF 解析、文本清洗、分块、embedding、ES9/Milvus 检索、rerank、LLM 生成
  -> 对外提供标准 API
```

也就是：

```text
Java 做业务主系统。
Python 做 AI/RAG 能力服务。
```

这样做的优势：

| 优势 | 说明 |
| --- | --- |
| 技术分工清晰 | Java 管业务，Python 管 AI 能力 |
| 开发效率高 | 文档解析、embedding、reranker 等能力用 Python 更快 |
| 企业集成友好 | Java 系统只需要调用 HTTP API，不需要理解内部 RAG 实现 |
| 后续可演进 | 如果某些能力稳定后，再考虑迁移到 Java |
| 风险低 | 不需要为了技术栈统一而牺牲 AI 生态 |

推荐演进路线：

```text
第一阶段：Python 脚本跑通吉野家5S手册 RAG Demo
第二阶段：封装成 Python FastAPI 服务
第三阶段：Java / Spring Boot 通过 HTTP 调用 RAG 服务
第四阶段：增加鉴权、日志、问题记录、反馈机制
第五阶段：根据企业要求，评估是否迁移部分检索逻辑到 Java
```

Python RAG 服务可以先提供一个核心接口：

```http
POST /ask
```

请求示例：

```json
{
  "question": "整理10分包含哪些检查项？",
  "mode": "hybrid",
  "top_k": 5
}
```

返回示例：

```json
{
  "answer": "整理10分包含物品清理、个人物品、物品分层存放等检查项。",
  "citations": [
    {
      "source": "吉野家5S管理手册.pdf",
      "page": 22,
      "section": "5S检查评估表"
    }
  ],
  "retrieved_chunks": [
    {
      "content": "...",
      "page": 22,
      "section": "5S检查评估表",
      "score": 0.82
    }
  ],
  "answerable": true
}
```

对于无法回答的问题，返回也要结构化：

```json
{
  "answer": "手册中未找到关于“李四”的信息。",
  "citations": [],
  "retrieved_chunks": [],
  "answerable": false
}
```

Java 侧只需要关心：

- 用户问题
- 调用 `/ask`
- 展示答案
- 展示引用页码
- 记录用户反馈
- 做权限和审计

后续如果进入企业级落地，可以再增加：

- `POST /documents/upload`：上传文档。
- `POST /documents/index`：触发文档解析和入库。
- `GET /documents/{id}/status`：查看索引状态。
- `POST /feedback`：记录答案好坏。
- `GET /health`：健康检查。

当前判断：

```text
不要急着纯 Java 化。
先用 Python 把 RAG 准确性做好，再用 Java 做系统集成。
Python RAG 服务 + Java 业务系统，是当前阶段最稳的企业落地路线。
```

### 5.9 关系型索引、向量索引与图数据库索引

索引的共同目标都是加快检索，但不同数据库的“检索问题”不一样。

关系型数据库索引主要加速：

```text
精确匹配 / 范围查询 / 排序
```

例如 MySQL：

```sql
WHERE user_id = 1001
WHERE age > 30
ORDER BY created_at DESC
```

它常用 B+Tree、Hash 等结构，目标是快速找到“满足条件的行”。

向量数据库索引主要加速：

```text
相似度搜索 / 最近邻搜索
```

例如 RAG 中的问题：

```text
找出和用户问题语义最相似的 Top-K 个 chunk。
```

它不是查：

```text
embedding = 某个固定值
```

而是查：

```text
哪些向量离 query_vector 最近。
```

常见向量索引包括：

- FLAT
- IVF
- HNSW
- DiskANN

向量索引通常是在速度、内存和召回率之间做权衡。很多向量索引是 ANN（Approximate Nearest Neighbor，近似最近邻），不一定保证 100% 找到理论上最近的向量。

可以这样对比：

| 对比点 | 关系型数据库索引 | 向量索引 | 图数据库索引 |
| --- | --- | --- | --- |
| 主要问题 | 找满足条件的数据 | 找语义最相似的数据 | 找节点、关系和路径 |
| 查询方式 | 精确匹配、范围查询、排序 | Top-K 相似度搜索 | 节点属性查询 + 图遍历 |
| 常见结构 | B+Tree、Hash | HNSW、IVF、FLAT、DiskANN | 属性索引、全文索引、向量索引、图遍历 |
| 结果含义 | 符合条件 | 最相似 | 有关系 / 有路径 |
| 典型问题 | `id=1`、`date > xxx` | “和这句话最像的文本有哪些” | “A 和 B 之间有什么关系” |

Neo4j 这样的图数据库也有索引，但它的核心能力不是普通相似度搜索，而是：

```text
节点 + 关系 + 路径
```

例如：

```text
(员工)-[:属于]->(门店)
(门店)-[:执行]->(5S检查项)
(检查项)-[:属于分类]->(整理)
```

Neo4j 中可以理解为几类能力：

| 类型 | 类比 | 作用 |
| --- | --- | --- |
| 属性索引 | MySQL 普通索引 | 快速按属性找节点，例如 `name='张三'` |
| 唯一约束 / 主键约束 | MySQL unique / primary key | 保证节点唯一，例如员工编号唯一 |
| 全文索引 | ES / BM25 | 按关键词搜索文本属性 |
| 向量索引 | Milvus / ES dense_vector | 查询相似 embedding |
| 图遍历 | 图数据库核心能力 | 沿关系查路径和关联 |

因此，Neo4j 不是简单替代 Milvus 或 ES，而是解决另一个问题：

```text
ES / BM25：关键词是否匹配。
Milvus / ES Vector：语义上像不像。
Neo4j：实体之间是什么关系。
```

这就进入 Graph RAG 的范畴。

对于 `吉野家5S管理手册`，如果只是问：

```text
5S 是什么？
整理有哪些要求？
5S 实施步骤有哪些？
```

普通 RAG 已经可以满足。

但如果后续要表达结构化关系，例如：

```text
检查项 属于 哪个一级分类
一级分类 对应 多少分
某检查项 违反后 扣几分
某岗位 负责 哪些区域
某区域 应该执行 哪些标准
```

图数据库就会有价值。

可以构造成类似知识图谱：

```text
(:Manual {name:"吉野家5S管理手册"})
  -[:包含章节]-> (:Section {name:"5S标准"})
  -[:包含检查表]-> (:Checklist)

(:ChecklistItem {name:"物品清理"})
  -[:属于分类]-> (:Category {name:"整理", score:"10分"})
  -[:扣分规则]-> (:Rule {text:"发现1处扣1分"})
```

这样用户问：

```text
整理 10 分下面有哪些检查项？
```

可以用图查询精准返回：

```cypher
MATCH (c:Category {name:"整理"})<-[:属于分类]-(item:ChecklistItem)
RETURN item
```

当前 demo 的推荐演进路线：

```text
第一阶段：ES9 + 向量 / BM25 混合检索
第二阶段：第 22 页检查表结构化成 JSON
第三阶段：如果要表达岗位、检查项、扣分规则、分类关系，再考虑 Neo4j
第四阶段：将 ES/Milvus 的语义检索与 Neo4j 的关系检索结合，形成 Graph RAG
```

一句话总结：

```text
关系型索引解决“精确找到谁”。
向量索引解决“谁和它最像”。
图数据库解决“谁和谁有什么关系”。
```

### 5.10 Milvus 与 ES9 的向量索引对比

Milvus 和 ES9 都能做向量检索，但它们的索引体系不是一一对应的。

Milvus 更像专业向量数据库，会把向量索引类型直接暴露出来供选择：

| Milvus 索引类型 | 核心思路 | 适合场景 |
| --- | --- | --- |
| FLAT | 暴力精确查找，计算 query 和所有向量的真实距离 | 数据量小、追求 100% 准确率 |
| IVF_FLAT / IVF_SQ8 / IVF_PQ | 先聚类分桶，再只搜索部分桶 | 大规模、高吞吐、可接受近似召回 |
| HNSW | 构建多层近邻图，沿图快速搜索 | 低延迟、高召回、内存相对充足 |
| DiskANN | 面向磁盘/SSD 的近邻图索引 | 数据量很大，无法全部放入内存 |

ES9 的向量检索则建立在 `dense_vector` 字段上。它主要通过 kNN 检索向量字段，并使用 HNSW 这类近似最近邻结构加速搜索。

ES9 常见思路可以这样理解：

| ES9 能力 | 类比 Milvus | 说明 |
| --- | --- | --- |
| `script_score` 暴力打分 | FLAT | 扫描候选文档并计算相似度，准确但慢，适合小数据或重排 |
| `dense_vector` + kNN | HNSW | ES 使用 HNSW 支持高效近似 kNN |
| `int8_hnsw` | HNSW + int8 量化 | 降低内存占用，牺牲少量精度 |
| `int4_hnsw` | HNSW + int4 量化 | 进一步降低内存，占用更少但精度损失更大 |
| `bbq_hnsw` | HNSW + binary quantization | 大幅降低内存，通常需要 oversampling/rerank 弥补精度 |
| `bbq_disk` | 类似磁盘型压缩向量索引 | 适合更大规模向量数据，减少内存压力 |

也就是说：

```text
Milvus：索引算法选择更丰富，偏专业向量数据库。
ES9：向量检索以 dense_vector/kNN 为入口，核心是 HNSW + 量化/磁盘优化。
```

ES9 的向量字段通常长这样：

```json
{
  "mappings": {
    "properties": {
      "content_vector": {
        "type": "dense_vector",
        "dims": 384,
        "similarity": "cosine",
        "index_options": {
          "type": "int8_hnsw"
        }
      }
    }
  }
}
```

其中几个关键参数：

| 参数 | 作用 |
| --- | --- |
| `dims` | 向量维度，必须和 embedding 模型输出维度一致 |
| `similarity` | 相似度计算方式，如 `cosine`、`dot_product`、`l2_norm` |
| `index_options.type` | 控制向量索引和量化方式，如 `int8_hnsw`、`int4_hnsw`、`bbq_hnsw`、`bbq_disk` |
| `index` | 是否为向量字段建索引，默认通常用于 kNN 检索 |

对当前 `吉野家5S管理手册` demo 来说，数据规模很小，不需要纠结复杂向量索引。更重要的是：

```text
先把解析、分块、embedding、metadata、混合检索、不知道机制做好。
```

如果使用 `BAAI/bge-small-zh-v1.5`，向量维度通常是 512；如果使用 `bge-m3`，向量维度是 1024。ES9 的 `dims` 必须和模型输出一致。

当前 demo 可以先选择：

```text
ES9 dense_vector + cosine similarity + 默认 kNN 索引策略
```

等数据量变大后，再考虑：

- 是否显式配置 `int8_hnsw`
- 是否使用 `bbq_hnsw` / `bbq_disk` 降低内存
- 是否使用 oversampling + rerank 提高量化后的召回质量
- 是否迁移到 Milvus / Qdrant 做更专业的向量检索

一句话总结：

```text
Milvus 更像专业向量检索引擎，索引类型选择多。
ES9 更像搜索引擎里加入向量能力，主要路线是 HNSW + 量化 + 混合检索。
```

### 5.11 KNN、ANN 与 Milvus 是否能替代 ES

Milvus 的检索能力已经不只是“基础向量相似度搜索”。它还支持很多增强检索方式，例如：

- 基础向量检索
- 过滤检索
- 范围检索
- 多向量 / 混合检索
- 分组检索
- metadata 标量过滤

所以在 RAG 场景里，如果核心需求是：

```text
文档 chunk 向量化
按语义召回 Top-K
结合 metadata 过滤
多路向量召回
做向量检索性能优化
```

Milvus 确实可以承担非常核心的检索角色，甚至可以替代 ES 的向量检索部分。

但“完全替代 ES”要看替代的是哪一部分能力。

| 能力 | Milvus | ES |
| --- | --- | --- |
| 向量相似度检索 | 很强，专业向量数据库 | 支持，但不是唯一核心 |
| 向量索引类型 | 丰富，适合大规模向量检索 | 主要围绕 dense_vector/kNN/HNSW/量化 |
| metadata 过滤 | 支持 | 支持，且搜索/过滤生态成熟 |
| BM25 全文检索 | 有 sparse/hybrid 能力，但传统全文搜索不是它的主战场 | 核心强项 |
| 中文分词、关键词搜索 | 不是主定位 | 生态成熟 |
| 聚合分析 | 有限 | 强项 |
| 日志检索 / 可观测性 | 不适合 | 强项 |
| 复杂搜索排序规则 | 偏向向量检索 | 更成熟 |
| RAG 向量库 | 很适合 | 也适合，尤其适合混合检索 demo |

所以更准确的判断是：

```text
如果系统核心是向量检索，Milvus 可以替代 ES 的向量检索角色。
如果系统还强依赖关键词搜索、复杂过滤、聚合、日志、搜索后台能力，ES 仍然有价值。
```

对于当前 `吉野家5S管理手册` demo：

```text
ES9：适合展示 BM25 + 向量的混合检索，贴近领导当前关注点。
Milvus：适合展示专业向量数据库能力，尤其是过滤检索、多向量检索、分组检索等。
```

后续可以把两者定位成：

```text
ES9：业务搜索入口，负责关键词、过滤、混合检索、展示和解释性。
Milvus：专业向量检索后端，负责大规模语义向量召回。
```

也可以在后续实验中做横向对比：

```text
同一批 chunk
同一个 embedding 模型
同一组问题
分别写入 ES9 和 Milvus
比较召回质量、延迟、配置复杂度和演示效果
```

#### KNN 与 ANN 的区别

KNN 是：

```text
K-Nearest Neighbors
K 个最近邻
```

它描述的是一个检索目标：

```text
给定一个 query vector，找到距离它最近的 K 个向量。
```

例如：

```text
找和“5S 实施步骤有哪些？”这个问题最相似的 5 个 chunk。
```

ANN 是：

```text
Approximate Nearest Neighbor
近似最近邻
```

它描述的是一种实现方式：

```text
不保证每次都找到理论上绝对最近的 K 个向量，
而是用近似算法快速找到足够接近的结果。
```

二者关系可以这样理解：

```text
KNN 是目标。
ANN 是为了更快实现 KNN 的近似方法。
```

精确 KNN：

```text
把 query vector 和库里所有向量逐个计算距离
排序
返回最近的 K 个
```

优点：

- 最准确
- 结果可解释

缺点：

- 数据量大时非常慢
- 计算成本高

ANN：

```text
通过 HNSW、IVF、DiskANN 等索引结构缩小搜索范围
快速找到近似最近的 K 个
```

优点：

- 速度快
- 适合大规模向量库

缺点：

- 可能漏掉理论上最相似的结果
- 需要调参数平衡速度和召回率

可以类比：

```text
精确 KNN：全班每个人都问一遍，找最合适的 5 个。
ANN：先按小组、关系网或区域快速定位候选人，再从候选人里找 5 个。
```

常见 ANN 索引：

| 索引 | 思路 |
| --- | --- |
| IVF | 先聚类分桶，查询时只查最相关的几个桶 |
| HNSW | 建多层近邻图，沿图快速搜索 |
| DiskANN | 把近邻图和磁盘访问结合，支持更大数据 |

在 Milvus 里：

```text
FLAT 更接近精确 KNN。
IVF / HNSW / DiskANN 属于 ANN 思路。
```

在 ES9 里：

```text
script_score 暴力计算更接近精确 KNN。
dense_vector + kNN 检索通常走 HNSW/量化等 ANN 路线。
```

对于 RAG 系统，通常不追求每次都做精确 KNN，而是用 ANN 先快速召回候选，再通过 reranker 或 LLM 上下文判断提高最终可靠性。

当前 demo 可以这样理解：

```text
小数据：精确检索和 ANN 体验差异不明显。
数据变大：ANN 是必要的，否则检索会慢。
高可靠：ANN 召回 + rerank + 不知道机制，比单纯追求向量最近更重要。
```

### 5.12 ES + Milvus 组合使用与 Milvus 索引粒度

ES 和 Milvus 不是非此即彼，它们可以组合使用。

更合理的分工是：

```text
ES：负责关键词、精确匹配、过滤、聚合、业务搜索。
Milvus：负责向量相似度检索。
LLM：负责基于召回结果生成答案。
```

例如用户问：

```text
整理 10 分包含哪些检查项？
```

可以设计为：

```text
ES 先做关键词 / 字段过滤：
  source = 吉野家5S管理手册
  chunk_type = table_row
  content 包含 整理 / 10分

Milvus 做语义召回：
  找和问题语义最相似的 chunk

融合层：
  合并 ES 和 Milvus 的候选结果
  rerank
  判断是否可回答
  交给 LLM 生成答案
```

这样可以结合两者优势：

```text
ES 擅长“查得准”。
Milvus 擅长“找得像”。
二者结合更稳。
```

#### 精确匹配、KNN 与 ANN 的区别

这里需要区分三个概念：

| 概念 | 解决的问题 | 示例 |
| --- | --- | --- |
| 精确匹配 | 字段值是否完全符合条件 | `source = '吉野家5S管理手册.pdf'` |
| KNN | 精确找出距离 query vector 最近的 K 个向量 | 暴力计算所有向量距离后排序 |
| ANN | 近似找出距离 query vector 最近的 K 个向量 | 用 HNSW、IVF、DiskANN 等索引快速召回 |

所以 KNN 不是字段精确匹配，而是：

```text
精确最近邻。
```

ANN 是：

```text
近似最近邻。
```

更准确的理解是：

```text
精确匹配：字段值必须对得上。
精确 KNN：向量距离精确最近。
ANN：向量距离近似最近，但速度更快。
```

实际系统通常不是单选，而是组合：

```text
metadata filter / BM25 先缩小范围
ANN 快速召回候选
reranker 做二次精排
LLM 做基于上下文回答
不知道机制兜底
```

这就是为什么真实业务系统会同时需要：

- 精确匹配
- 关键词检索
- 向量检索
- rerank
- 拒答机制

#### Milvus 的索引粒度

Milvus 的索引通常是按字段建立的，不是按单条数据建立的。

例如一个 Collection：

```text
collection: rag_chunks
fields:
  id
  content
  dense_vector
  sparse_vector
  image_vector
  source
  page
  chunk_type
```

可以给不同字段配置不同索引：

```text
dense_vector -> HNSW
sparse_vector -> sparse index
image_vector -> IVF / HNSW
source/page/chunk_type -> 标量索引
```

但通常不能做到：

```text
第 1 条 entity 用 HNSW
第 2 条 entity 用 FLAT
第 3 条 entity 用 DiskANN
```

也就是说：

```text
Milvus 的索引粒度主要是字段级，不是单条数据级。
```

Partition 的作用也不是给每条数据配置不同索引，而是把同一个 Collection 中的数据按逻辑分区，查询时缩小搜索范围。

例如：

```text
collection: rag_chunks
partition: yoshinoya_5s
partition: food_safety
partition: employee_training
```

当用户只问 5S 手册时，可以只搜：

```text
partition = yoshinoya_5s
```

这样可以减少无关数据干扰，也能提升查询效率。

当前 demo 的推荐设计：

```text
ES9：
  content 做 BM25
  source/page/section/chunk_type 做过滤
  适合关键词和精确条件

Milvus：
  dense_vector 做语义召回
  source/page/chunk_type 也保留 metadata
  适合向量相似度搜索

融合层：
  合并 ES 和 Milvus 结果
  rerank
  判断是否可回答
```

一句话总结：

```text
ES + Milvus 可以结合用。
KNN/ANN 是向量最近邻问题，不是字段精确匹配。
Milvus 索引通常按字段建，不是按单条数据建。
Partition 用来缩小检索范围，不是给每条数据配置不同索引。
```
