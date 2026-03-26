---
title: KGoT Python Executor
emoji: 🐍
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
---

Remote Python executor for Knowledge Graph of Thoughts with comprehensive ML/data science capabilities.

## Hardware
- **CPU Basic** (Free): 2 vCPU, 16GB RAM, 50GB ephemeral disk

## Included Libraries

### Core Data
- numpy, pandas

### NLP
- nltk, spacy, transformers, textblob, gensim, fastembed

### Vision
- pillow, opencv-python-headless, torchvision, scikit-image

### Advanced Data
- polars, duckdb, sktime

### Time Series & Graph
- torch (CPU), torch_geometric, networkx
- pyg_lib, torch_scatter, torch_sparse, torch_cluster

### Audio
- pydub, soundfile, librosa

### Documents
- python-docx, csvkit, rich, svgwrite, svgpathtools, PyMuPDF

### Utilities
- pydantic, tqdm, ipython, click, tenacity, ujson, msgpack, xxhash, zstandard, httpx

### ML
- scikit-learn, joblib

### Visualization
- matplotlib, seaborn

## API Endpoints

- `GET /` - Basic service info
- `GET /health` - Health check
- `POST /run` - Code execution endpoint

### POST /run
```python
{
   "required_modules": ["list", "of", "packages"],  # Optional - auto-install
   "code": "python code to execute"
}
```

Returns:
```python
{
    "output": "execution output"
}
```

## Timeout
- Maximum execution time: 240 seconds