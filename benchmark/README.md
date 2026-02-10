# Caliby Benchmarks

Comprehensive benchmarks comparing Caliby against other popular vector search libraries.

## Available Benchmarks

### 1. Multi-Database Comparison (Caliby vs ChromaDB vs Qdrant vs Weaviate)
**Full vector database comparison** - Tests document storage, retrieval, vector search, filtered search, and hybrid search
- **Script**: `compare_vectordb.py`
- **Documentation**: [CHROMADB_BENCHMARK.md](CHROMADB_BENCHMARK.md)
- **What's tested**: Insertion throughput, document retrieval, vector search, filtered search, hybrid search, storage efficiency
- **Dataset**: SIFT1M (1M vectors, 128 dimensions)
- **Databases**: Caliby (your implementation), ChromaDB, Qdrant, Weaviate (all local/embedded)

```bash
# Install dependencies
pip install chromadb qdrant-client weaviate-client

# Quick start - Compare all databases
python compare_vectordb.py --num-vectors 100000

# Compare specific databases
python compare_vectordb.py --caliby-only --num-vectors 100000
python compare_vectordb.py --chromadb-only --num-vectors 100000
python compare_vectordb.py --qdrant-only --num-vectors 100000
python compare_vectordb.py --weaviate-only --num-vectors 100000
```

### 2. HNSW Comparison
Compare HNSW implementations across Caliby, Usearch, and Faiss
- **Script**: `compare_hnsw.py`
- **Dataset**: SIFT1M

### 3. DiskANN Comparison
Compare DiskANN implementations
- **Script**: `compare_diskann.py`

### 4. IVF+PQ Comparison
Compare quantization-based indexes with FAISS
- **Script**: `compare_ivfpq_faiss.py`

## IVF+PQ Benchmark

### FAISS IVF+PQ Baseline Results (SIFT1M)

**Configuration:**
- Dataset: 1M SIFT vectors (128 dimensions)  
- Clusters: 256
- Subquantizers: 8
- PQ codes: 256 (8 bits)

**Performance:**
- Build time: 9.98s (126k vectors/sec)
- Index size: ~7.6 MB compressed
- Recall@10 (nprobe=16): 35.6%
- QPS (nprobe=16): 52,754
- Latency (nprobe=16): 0.019 ms/query

**Recall vs nprobe:**
```
nprobe    Recall@10    QPS
1         0.262        447,140
4         0.340        164,726
16        0.356        52,754
32        0.356        28,023  
64        0.356        18,465
128       0.356        10,372
```

Run with: `python3 faiss_ivfpq_baseline.py`

### Caliby IVF+PQ Status

✅ **Fixed:** Codebook overflow bug - can now correctly handle multi-page codebooks
⚠️  **Known Issues:** Search segfault with large datasets (>50k vectors)  
📊 **Early Results:** ~40% recall@10 (better than FAISS 35.6%!)

## HNSW Benchmark

Compare HNSW implementations across Caliby, Usearch, and Faiss using the SIFT1M dataset.

### Requirements

```bash
# Install benchmark dependencies
pip install usearch faiss-cpu numpy

# Or install all at once
pip install usearch faiss-cpu numpy
```

### Usage

Run the full benchmark:
```bash
python compare_hnsw.py
```

Customize parameters:
```bash
python compare_hnsw.py --M 32 --ef-construction 400 --ef-search 100
```

Skip specific libraries:
```bash
python compare_hnsw.py --skip usearch faiss  # Only benchmark caliby
```

Specify data directory:
```bash
python compare_hnsw.py --data-dir /path/to/sift1m
```

### Command-line Options

- `--data-dir`: Directory for SIFT1M dataset (default: `./sift1m`)
- `--M`: HNSW M parameter - controls connectivity (default: 16)
- `--ef-construction`: Construction time search depth (default: 200)
- `--ef-search`: Query time search depth (default: 50)
- `--skip`: Libraries to skip (choices: `caliby`, `usearch`, `faiss`)

### Metrics

The benchmark measures:

1. **Build Time**: Time to construct the index
2. **Index Size**: Disk/memory footprint in MB
3. **QPS**: Queries per second (throughput)
4. **Latency**: P50, P95, P99 percentiles in milliseconds
5. **Recall@10**: Accuracy compared to ground truth

### Dataset

The benchmark uses [SIFT1M](http://corpus-texmex.irisa.fr/):
- 1,000,000 base vectors (128 dimensions)
- 10,000 query vectors
- Ground truth for Recall@k computation

The dataset is automatically downloaded on first run (~200MB compressed).

### Example Output

```
==================================================================================================
BENCHMARK RESULTS SUMMARY
==================================================================================================

Library         Build(s)     Size(MB)     QPS          P50(ms)    P95(ms)    P99(ms)    Recall@10   
----------------------------------------------------------------------------------------------------
Usearch         45.23        345.12       8234.5       0.121      0.234      0.456      0.9823      
Faiss           52.18        389.45       7891.2       0.127      0.245      0.501      0.9845      
Caliby          48.76        356.78       7654.3       0.131      0.252      0.523      0.9801      

==================================================================================================

WINNERS:
  🏆 Highest QPS:        Usearch (8234.5 queries/sec)
  🏆 Best Recall@10:     Faiss (0.9845)
  🏆 Lowest P50 Latency: Usearch (0.121 ms)
  🏆 Smallest Index:     Usearch (345.12 MB)
  🏆 Fastest Build:      Usearch (45.23 s)

==================================================================================================
```

### Interpreting Results

- **QPS**: Higher is better - measures search throughput
- **Latency**: Lower is better - measures per-query response time
- **Recall@10**: Higher is better (max 1.0) - measures accuracy
- **Build Time**: Lower is better - time to construct index
- **Index Size**: Lower is better - storage efficiency

### Notes

- First run downloads SIFT1M dataset (~200MB)
- Benchmark includes warm-up phase before measurements
- All libraries use L2 distance metric
- Results may vary based on hardware and configuration

## Cohere Datasets

The benchmark now supports Cohere Wikipedia embedding datasets (768-dimensional vectors).

### Available Datasets

- **cohere1m**: 1 million vectors, 768 dimensions, 10K queries
- **cohere10m**: 10 million vectors, 768 dimensions, 10K queries

### Creating Datasets

Use the provided script to create synthetic Cohere-like datasets:

```bash
# Create Cohere-1M dataset (~3GB)
python3 create_cohere_datasets.py --dataset cohere1m

# Create Cohere-10M dataset (~30GB, takes 30-60 minutes)
python3 create_cohere_datasets.py --dataset cohere10m

# Create both
python3 create_cohere_datasets.py --dataset both
```

The datasets will be created at: `~/Workspace/datasets/cohere/`

### Running Benchmarks

```bash
# Benchmark with Cohere-1M
python3 compare_vectordb.py --dataset cohere1m --caliby-only

# Benchmark with Cohere-10M (subset)
python3 compare_vectordb.py --dataset cohere10m --caliby-only --num-vectors 100000

# Full Cohere-10M benchmark
python3 compare_vectordb.py --dataset cohere10m --caliby-only
```

### Real Cohere Datasets

For real Cohere Wikipedia embeddings:
- HuggingFace: https://huggingface.co/datasets/Cohere/wikipedia-22-12-en-embeddings
- 35M+ Wikipedia articles with 768-dim embeddings from Cohere's embedding model
