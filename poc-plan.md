# PoC Plan: caliby

## Project Classification
- **Type:** infrastructure (embeddable library / SDK)
- **Key Technologies:** C++17, Python (pybind11 bindings), CMake, NumPy, SIMD intrinsics
- **ODH Relevance:** Caliby is a high-performance embeddable vector database library relevant to the AI/ML ecosystem. It can serve as the vector storage backend for RAG pipelines, semantic search, agentic memory stores, and recommendation systems — all common patterns in Open Data Hub workloads. Validating it on OpenShift proves it builds and runs correctly in containerized Linux environments.

## PoC Objectives
What we want to prove:
1. The C++ extension compiles correctly inside a container (requires CMake, pybind11, C++17 compiler, and potentially SIMD support)
2. The Python `caliby` module can be imported and used for core operations
3. Collections can be created with configurable dimensions and index types
4. Vectors can be inserted, searched, and deleted correctly
5. The library's test suite passes in a containerized environment, confirming buffer pool management, persistence, and distance computations work

## Infrastructure Requirements
- **Inference Server:** none
- **Vector Database:** none (Caliby IS a vector database library)
- **Embedding Model:** none
- **GPU Required:** no
- **Persistent Storage:** none (tests use temp directories)
- **Resource Profile:** medium (C++ compilation is CPU/memory intensive; tests exercise buffer pools)
- **Sidecar Containers:** none

## Test Scenarios

### Scenario 1: Import Check
- **Description:** Verify the caliby C++ extension module can be imported successfully after building
- **Type:** cli
- **Input:** `python -c "import caliby; print('Caliby imported successfully'); print(dir(caliby))"`
- **Expected:** Job exits 0, prints module attributes confirming the native extension loaded
- **Timeout:** 30 seconds

### Scenario 2: Collection Create, Insert, and Search
- **Description:** End-to-end test: create a Calico database instance, create a collection, insert 100 random vectors, and perform a top-k similarity search
- **Type:** cli
- **Input:** Python script that creates a collection, inserts vectors, and searches
- **Expected:** Job exits 0, search returns results, prints PASS message
- **Timeout:** 60 seconds

### Scenario 3: Initialization Tests
- **Description:** Run the `test_initialization.py` test suite to validate core database setup and configuration
- **Type:** cli
- **Input:** `python -m pytest tests/test_initialization.py -v --timeout=120 -x`
- **Expected:** All tests pass
- **Timeout:** 180 seconds

### Scenario 4: Collection Basic Tests
- **Description:** Run the `test_collection.py` test suite to validate CRUD operations, vector search, metadata filtering
- **Type:** cli
- **Input:** `python -m pytest tests/test_collection.py -v --timeout=120 -x`
- **Expected:** All tests pass confirming add/search/delete/update operations work correctly
- **Timeout:** 180 seconds

### Scenario 5: Distance Metric Tests
- **Description:** Run distance metric tests to validate cosine, L2, inner product similarity computations (including SIMD paths)
- **Type:** cli
- **Input:** `python -m pytest tests/test_distance.py -v --timeout=60 -x`
- **Expected:** All distance metric tests pass
- **Timeout:** 120 seconds

## Dockerfile Considerations
This is a C++ library with Python bindings — it is NOT a server. The Dockerfile must:

- **Base image:** Use a Python 3.11+ image with build tools (gcc/g++ >= 10 for C++17, CMake >= 3.16)
- **Build dependencies:** Install `cmake`, `g++`, `make`, and `pybind11` (included in `third_party/pybind11/`)
- **Build step:** Run `pip install .` which triggers CMake via setuptools to compile the C++ extension
- **Runtime dependencies:** Only `numpy` is needed at runtime
- **Multi-stage build recommended:** Build stage compiles the extension; runtime stage copies the built wheel to keep the image small
- **ENTRYPOINT should be:** `python -c "import caliby; print('Caliby ready')"` or just `python`
- **CMD should default to:** `--help` or a version check
- **Do NOT add EXPOSE** — there is no port to expose. This is a library, not a server.
- **Do NOT install pytest in the runtime image** — install it only if tests need to run in the container (which they do for PoC validation, so include `pytest` and `pytest-benchmark` in the image)
- **Note on SIMD:** The C++ code uses SIMD intrinsics. The container should be built for x86_64 with SSE/AVX support. Add `-march=native` or appropriate flags during compilation if the build system doesn't already do so.

## Deployment Considerations
- **Deployment model: Job** — Caliby is a library, not a long-running server. Do NOT deploy as a Deployment — it has no server loop, no listening port, and the container will exit immediately, causing CrashLoopBackOff.
- **Do NOT create a Service** — there is no port to expose.
- **Testing:** Each test scenario runs as a Kubernetes Job. The Job executes a CLI command, and success is determined by the Job's exit code (0 = pass) and output captured via `kubectl logs`.
- **Resource requests:** 1Gi RAM, 500m CPU minimum. The C++ buffer pool tests may allocate significant memory. 2Gi RAM recommended for the test suite.
- **No external dependencies:** Caliby has zero runtime infrastructure dependencies — no databases, no API keys, no network access needed. This makes it ideal for isolated Job-based testing.