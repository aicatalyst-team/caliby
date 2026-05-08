<p align="center">
  <a href="README.md">English</a> | <strong>中文</strong>
</p>

<p align="center">
 <img src="docs/static/caliby_logo.png" alt="Caliby Logo" width="50%">
</p>

<p align="center">
  <a href="https://pypi.org/project/caliby/"><img src="https://img.shields.io/pypi/v/caliby.svg?logo=pypi" alt="PyPI Version"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License"></a>
  <a href="README.md"><img src="https://img.shields.io/badge/README-EN-blue.svg" alt="English README"></a>
</p>

<p align="center">
  <em><sub>⭐ 如果 Caliby 对你有用，请给一个 Star！</sub></em>
</p>

---

Caliby 是一个**可嵌入的向量数据库**，专为需要突破内存限制而又不想引入分布式系统复杂度的 AI 应用而设计。与需要独立服务端基础设施的客户端-服务器向量数据库不同，Caliby 直接在你的应用程序进程内运行，提供文档存储、向量搜索和过滤搜索等丰富功能。

## 🎯 为什么选择 Caliby？

### 现有方案的局限

| 方案 | 局限性 |
|----------|------------|
| **HNSWLib / Faiss / Usearch** | 纯内存方案——数据超过 RAM 时崩溃或严重变慢 |
| **Pinecone / Weaviate / Qdrant** | 需要独立的服务端基础设施，引入网络延迟和运维负担 |
| **ChromaDB / LanceDB** | 索引选项有限，缺乏高效的超内存缓冲池 |

### Caliby 的做法：可嵌入 + 高性能 + 超内存扩展

Caliby 将**嵌入式库的简洁性**与**磁盘存储的扩展能力**相结合，同时在数据可放入内存时保持内存级向量搜索速度：

- **🔌 零基础设施**: `pip install caliby` —— 无需 Docker、无需服务端、无需配置
- **📦 随应用分发**: 将 Caliby 直接嵌入桌面应用、边缘设备或微服务
- **💾 单机承载十亿级向量**: 通过智能缓冲区管理，处理远超 RAM 的数据集
- **⚡ 内存级性能**: 当数据可放入 RAM 时，性能媲美甚至超越 HNSWLib/Faiss
- **📉 平滑降级**: 随着数据增长超过 RAM，性能平滑下降——而非灾难性崩溃

### 适用场景

- **🤖 AI Agent** —— 持久化记忆，支持重启恢复，随对话历史扩展
- **📱 桌面/移动应用** —— 本地优先的语义搜索，无需云依赖
- **🔧 开发者工具** —— 在 IDE 和 CLI 中嵌入代码搜索、文档检索
- **🏭 边缘计算** —— 在资源受限、无网络的设备上运行
- **🧪 快速原型验证** —— 数分钟内从想法到可工作的 RAG 管线

## ✨ 核心特性

- **🔌 可嵌入**: 单进程库，在你的应用程序内存空间中运行
- **💾 超内存扩展**: 创新的缓冲区池设计，处理 10-100 倍于 RAM 的数据集
- **📚 文档存储**: 灵活的 Schema，存储向量、文本和元数据
- **🔍 过滤搜索**: 高效的向量搜索与元数据过滤
- **🔗 混合搜索**: 结合向量相似度和 BM25 全文搜索
- **🔥 内存级速度**: 数据在 RAM 内时，性能匹配 HNSWLib/Faiss
- **🎯 多索引类型**: HNSW、DiskANN、IVF+PQ、B+树和倒排索引

## 📱 应用场景

| 场景 | 为什么用 Caliby？ | 示例 |
|----------|-------------|---------|
| **🤖 Agent 记忆存储** | 持久化的 Agent 记忆，无界增长，支持重启，无需外部数据库 | [agentic_memory_store.py](examples/agentic_memory_store.py) |
| **📚 RAG 管线** | 本地索引数百万文档片段，无 API 延迟的混合搜索 | [rag_pipeline.py](examples/rag_pipeline.py) |
| **🛒 推荐系统** | 随应用分发推荐能力，在边缘设备上离线工作 | [recommendation_system.py](examples/recommendation_system.py) |
| **🔍 语义搜索** | 桌面应用、开发者工具的本地优先搜索，支持离线 | [semantic_search.py](examples/semantic_search.py) |
| **🖼️ 图像相似度** | 嵌入照片应用的视觉搜索，无需上传云端 | [image_similarity_search.py](examples/image_similarity_search.py) |

## 🚀 快速开始

### 安装

**从 PyPI 安装（推荐）:**
```bash
pip install caliby
```

**从源码安装:**
```bash
git clone --recursive https://github.com/zxjcarrot/caliby.git
cd caliby
pip install -e .
```

**注意:** 需要 `--recursive` 标志来初始化 pybind11 子模块。如果你已经克隆但没有加此参数，运行：
```bash
git submodule update --init --recursive
```

### Collection API（推荐）

Collection API 提供了存储带向量、文本和元数据的文档的高级接口：

```python
import caliby
import numpy as np

# 初始化
caliby.set_buffer_config(size_gb=1.0)
caliby.open('/tmp/my_database', cleanup_if_exist=True)

# 定义 Schema
schema = caliby.Schema()
schema.add_field("category", caliby.FieldType.STRING)
schema.add_field("price", caliby.FieldType.FLOAT)

# 创建 128 维向量的 Collection
collection = caliby.Collection("products", schema, vector_dim=128)

# 添加文档（返回分配的 ID）
contents = ["带降噪功能的无线耳机",
            "适合越野跑的跑步鞋"]
metadatas = [{"category": "electronics", "price": 99.99},
             {"category": "sports", "price": 79.99}]
vectors = np.random.rand(2, 128).astype(np.float32).tolist()

doc_ids = collection.add(contents, metadatas, vectors)

# 创建索引
collection.create_hnsw_index("vec_idx", M=16, ef_construction=200)
collection.create_text_index("text_idx")
collection.create_metadata_index("category_idx", ["category"])

# 带元数据过滤的向量搜索
query = np.random.rand(128).astype(np.float32)
results = collection.search_vector(query, "vec_idx", k=10,
                                   filter='{"category": {"$eq": "electronics"}}')

# 混合搜索（向量 + 文本）
results = collection.search_hybrid(query, "vec_idx",
                                   "无线耳机", "text_idx", k=10)

for r in results:
    print(f"文档 {r.doc_id}: 相关度={r.score:.4f}")

caliby.close()
```

📖 完整文档请参见 **[docs/COLLECTION_API.md](docs/COLLECTION_API.md)**，包含高级过滤、最佳实践和性能调优。

### 底层索引 API

如需直接控制索引：

```python
import caliby
import numpy as np

# 初始化系统并配置缓冲池
caliby.set_buffer_config(size_gb=1.0)
caliby.open('/tmp/caliby_data')

# 创建 HNSW 索引
index = caliby.HnswIndex(
    max_elements=1_000_000,
    dim=128,
    M=16,
    ef_construction=200,
    enable_prefetch=True,
    skip_recovery=False,
    index_id=0,
    name='user_embeddings',
)

# 批量添加向量
vectors = np.random.rand(10000, 128).astype(np.float32)
index.add_points(vectors, num_threads=4)

# 获取索引信息
print(f"索引名称: {index.get_name()}")
print(f"向量维度: {index.get_dim()}")

# 单查询搜索
query = np.random.rand(128).astype(np.float32)
labels, distances = index.search_knn(query, k=10, ef_search_param=50)

# 批量并行搜索
queries = np.random.rand(100, 128).astype(np.float32)
results = index.search_knn_parallel(queries, k=10, ef_search_param=50, num_threads=4)

caliby.close()
```

## 🏗️ 索引类型

### HNSW（分层可导航小世界图）

最适合：高召回率要求，中等至大规模数据集

```python
import caliby
import numpy as np

caliby.set_buffer_config(size_gb=2.0)
caliby.open('/tmp/caliby_data')

index = caliby.HnswIndex(
    max_elements=1_000_000,
    dim=128,
    M=16,                    # 越高 = 召回越好，但内存占用更多
    ef_construction=200,     # 越高 = 图质量越好，但构建越慢
    enable_prefetch=True,
    skip_recovery=False,
    index_id=0,
    name='my_vectors',
)

vectors = np.random.rand(100000, 128).astype(np.float32)
index.add_points(vectors, num_threads=4)

query = np.random.rand(128).astype(np.float32)
labels, distances = index.search_knn(query, k=10, ef_search_param=100)
```

### DiskANN（Vamana 图）

最适合：带过滤的搜索，动态更新，带标签的超大图

```python
import caliby
import numpy as np

caliby.set_buffer_config(size_gb=2.0)
caliby.open('/tmp/caliby_data')

index = caliby.DiskANN(
    dimensions=128,
    max_elements=1_000_000,
    R_max_degree=64,    # 最大图度数 (R)
    is_dynamic=True     # 启用动态增删
)

vectors = np.random.rand(100000, 128).astype(np.float32)
tags = [[i % 100] for i in range(100000)]

params = caliby.BuildParams()
params.L_build = 100
params.alpha = 1.2
params.num_threads = 4

index.build(vectors, tags, params)

# 带过滤的搜索
search_params = caliby.SearchParams(L_search=50)
search_params.beam_width = 4

query = np.random.rand(128).astype(np.float32)
labels, distances = index.search(query, K=10, params=search_params)

# 标签过滤搜索
labels, distances = index.search_with_filter(query, filter_label=42, K=10, params=search_params)

# 动态操作
new_point = np.random.rand(128).astype(np.float32)
index.insert_point(new_point, tags=[99], external_id=100000)
index.lazy_delete(external_id=100000)
index.consolidate_deletes(params)
```

### IVF+PQ（倒排文件 + 乘积量化）

最适合：超大规模数据集（千万级以上），内存受限环境

```python
import caliby
import numpy as np

caliby.set_buffer_config(size_gb=0.5)
caliby.open('/tmp/caliby_data')

index = caliby.IVFPQIndex(
    max_elements=10_000_000,
    dim=128,
    num_clusters=256,
    num_subquantizers=8,
    retrain_interval=10000,
    skip_recovery=False,
    index_id=0,
    name='large_dataset'
)

# 先训练索引（IVF+PQ 必需）
training_data = np.random.rand(50000, 128).astype(np.float32)
index.train(training_data)

# 训练完成后添加向量
vectors = np.random.rand(1000000, 128).astype(np.float32)
index.add_points(vectors, num_threads=4)

query = np.random.rand(128).astype(np.float32)
labels, distances = index.search_knn(query, k=10, nprobe=8)
```

## 🔧 高级配置

### 持久化与恢复

```python
import caliby

# 索引通过缓冲池自动持久化
caliby.set_buffer_config(size_gb=1.0)
caliby.open('/path/to/caliby_data')

# 创建索引（自动持久化）
index = caliby.HnswIndex(
    max_elements=1_000_000,
    dim=128,
    M=16,
    ef_construction=200,
    enable_prefetch=True,
    skip_recovery=False,  # 设为 False 以启用恢复
    index_id=1,
    name='my_index'
)

# 手动刷盘确保所有数据写入
index.flush()

# 重新打开时自动恢复
caliby.close()

# 之后：重新打开并恢复
caliby.open('/path/to/caliby_data')
recovered_index = caliby.HnswIndex(
    max_elements=1_000_000,
    dim=128,
    M=16,
    ef_construction=200,
    enable_prefetch=True,
    skip_recovery=False,  # 将恢复已有索引
    index_id=1,           # 必须与原来一致
    name='my_index'
)

if recovered_index.was_recovered():
    print("索引从磁盘成功恢复！")
```

## 📁 项目结构

```
caliby/
├── include/caliby/          # C++ 头文件
│   ├── calico.hpp           # 核心缓冲池系统
│   ├── hnsw.hpp             # HNSW 索引
│   ├── ivfpq.hpp            # IVF+PQ 索引
│   ├── diskann.hpp          # DiskANN 索引
│   ├── catalog.hpp          # 索引目录管理
│   └── distance.hpp         # 距离函数
├── src/                     # C++ 实现
│   ├── bindings.cpp         # Python 绑定
│   ├── hnsw.cpp
│   ├── ivfpq.cpp
│   └── calico.cpp
├── examples/                # 使用示例
├── benchmark/               # 性能基准测试
├── tests/                   # Python 测试
└── third_party/             # 第三方依赖
    └── pybind11/            # Python 绑定库（子模块）
```

## 🛠️ 从源码构建

### 系统依赖

Caliby 需要以下系统依赖：
- C++17 兼容的编译器（GCC 9+ 或 Clang 10+）
- CMake 3.15+
- OpenMP
- Abseil C++ 库
- Python 3.8+

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake libomp-dev libabsl-dev python3-dev
```

**Fedora/RHEL:**
```bash
sudo dnf install -y gcc-c++ cmake libomp-devel abseil-cpp-devel python3-devel
```

### 构建

```bash
git clone https://github.com/zxjcarrot/caliby.git
cd caliby
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# 安装 Python 包
cd ..
pip install -e .
```

### 运行测试

```bash
# C++ 测试
cd build && ctest --output-on-failure

# Python 测试
pytest python/tests/
```

## 📚 文档

- **[Collection API 指南](docs/COLLECTION_API.md)** —— 包含向量、文本和元数据的文档高级 API
- **[使用指南](docs/USAGE.md)** —— 通用使用模式和示例
- **[性能基准](benchmark/README.md)** —— 性能对比和基准测试工具

## 🔬 Caliby 如何实现超内存处理

与数据超过内存就崩溃或卡死的纯内存库不同，Caliby 采用**数据库风格的缓冲池**：

```
┌─────────────────────────────────────────────────────────────────┐
│                         你的应用                                │
├─────────────────────────────────────────────────────────────────┤
│                     Caliby（嵌入式库）                          │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    缓冲池 (RAM)                          │   │
│  │   ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐      │   │
│  │   │热页 │ │热页 │ │温页 │ │温页 │ │冷页 │ │冷页 │ ...  │   │
│  │   └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘      │   │
│  └─────────────────────────────────────────────────────────┘   │
│                            ▲  │                                 │
│                    淘汰冷页 │  │ 访问时并行预取                   │
│                            │  ▼                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                     磁盘存储                             │   │
│  │        (SSD/NVMe)                                        │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

**核心洞察**：大多数向量搜索工作负载具有**局部性**——最近访问的向量很可能再次被访问。Caliby 利用这一点，将热数据保留在 RAM 中，并将冷数据无缝换出到磁盘。

| 数据量 vs 内存 | Caliby 表现 |
|-----------------|-----------------|
| 数据 < 内存 | 🚀 完全内存级速度（匹配 HNSWLib） |
| 数据 ≈ 内存 | ⚡ 大部分在内存中，偶发磁盘读取 |
| 数据 >> 内存 | 💾 工作组在内存中，平滑的磁盘访问 |

## 🤝 参与贡献

我们欢迎贡献！详情请参见 [贡献指南](CONTRIBUTING.md)。

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m '添加了某特性'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 提交 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 —— 详见 [LICENSE](LICENSE) 文件。

## 📬 联系方式

- **Issues**: [GitHub Issues](https://github.com/zxjcarrot/caliby/issues)
- **Discussions**: [GitHub Discussions](https://github.com/zxjcarrot/caliby/discussions)
- **Email**: xinjing@mit.edu

---

<p align="center">
  <em>Caliby —— AI 时代的可嵌入向量数据库</em>
</p>
