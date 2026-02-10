# Edge Inference VLM

A privacy-preserving vision-language system that captures your screen, encodes it locally through CLIP, and generates text descriptions via a server-side LLM. **Only abstract embedding vectors leave the device — never raw images.**

## Architecture

```
Edge Client (privacy boundary)              Server
┌─────────────────────────┐    HTTP/JSON  ┌────────────────────────────┐
│ Screen Capture (mss)    │               │ FastAPI                    │
│        ↓                │               │   ↓                        │
│ CLIP ViT-L/14-336       │  (576,1024)   │ PyTorch Projector (40 MB)  │
│ (HuggingFace, MPS)      │ ────────────► │   ↓ (576,4096)             │
│        ↓                │  float16 b64  │ ctypes embedding inject    │
│ (576,1024) float16 emb  │               │   ↓                        │
│                         │ ◄──────────── │ Vicuna-7B GGUF (4 GB Q4)   │
│ Privacy: only abstract  │  description  │   ↓                        │
│ vectors leave device    │               │ SQLite DB                  │
└─────────────────────────┘               └────────────────────────────┘
```

**Privacy model:** The CLIP vision tower runs entirely on the edge device. Only float16 embedding vectors (abstract, non-reconstructable) are transmitted to the server. The server never sees the original screenshots.

**Hybrid backend:** The server uses a lightweight PyTorch projector (~40 MB) to map CLIP embeddings into the LLM's space, then injects them directly into a GGUF quantized LLM via llama.cpp's C API. This keeps memory under 5 GB while achieving ~5-10 tokens/second.

## Performance (Apple Silicon M3, 18 GB)

| Metric | Value |
|--------|-------|
| Server memory | ~4.5 GB (GGUF Q4 + projector) |
| Edge CLIP memory | ~0.6 GB |
| Server load time | ~25 seconds |
| CLIP encode time | ~0.5 seconds |
| LLM inference | ~5-7 seconds per frame |
| Embedding payload | ~1.5 MB (float16, 576x1024) |

## Setup

### Prerequisites

- Python 3.10+
- macOS with Apple Silicon (M1/M2/M3) for Metal GPU acceleration

### Install

```bash
cd edge-inference
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

For Metal GPU support with llama.cpp:

```bash
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/metal
```

### Download Models & Extract Projector

```bash
# Download the GGUF quantized LLM (~3.8 GB)
python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download('mys/ggml_llava-v1.5-7b', 'ggml-model-q4_k.gguf', local_dir='models')
"

# Extract projector weights from HuggingFace LLaVA (~40 MB)
# Downloads the full model once, extracts only the projector, then frees memory
python3 scripts/extract_projector.py
```

## Usage

### 1. Start the Server

```bash
source venv/bin/activate
python3 -m server.main
```

### 2. Start the Edge Client

In a separate terminal:

```bash
source venv/bin/activate
python3 -m edge.main
```

The client loads CLIP (~3s), waits for the server, then begins capturing and encoding screenshots.

### CLI Options

**Server:**
```
--host HOST        Bind address (default: 127.0.0.1)
--port PORT        Bind port (default: 8000)
--model PATH       Path to GGUF LLM model
--projector PATH   Path to projector weights (.pt)
```

**Edge Client:**
```
--interval SEC     Capture interval in seconds (default: 1.0)
--monitor IDX      Monitor index to capture (default: 1)
--server-url URL   Server URL (default: http://127.0.0.1:8000)
```

### Environment Variables

All config values can be overridden with `EDGE_` prefix:

```bash
EDGE_CAPTURE_INTERVAL_SECONDS=2.0
EDGE_VISION_MODEL_ID=openai/clip-vit-large-patch14-336
EDGE_SERVER_PORT=9000
EDGE_N_GPU_LAYERS=-1
EDGE_MAX_NEW_TOKENS=512
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Server status and model readiness |
| POST | `/api/v1/embeddings` | Submit CLIP embeddings for description |
| GET | `/api/v1/descriptions` | List descriptions (paginated) |
| GET | `/api/v1/descriptions/{frame_id}` | Get a specific description |

## Project Structure

```
edge-inference/
├── config.py                # Centralized configuration
├── requirements.txt         # Python dependencies
├── scripts/
│   └── extract_projector.py # One-time projector weight extraction
├── models/                  # Model files (downloaded/extracted)
│   ├── ggml-model-q4_k.gguf    # Vicuna-7B GGUF (4 GB)
│   ├── projector_fp16.pt       # Extracted projector weights (40 MB)
│   └── projector_config.json   # Projector architecture config
├── shared/
│   ├── __init__.py
│   └── schemas.py           # Pydantic API schemas
├── edge/
│   ├── __init__.py
│   ├── capture.py           # Screen capture (mss)
│   ├── vision.py            # CLIP vision tower (HuggingFace)
│   ├── client.py            # Embedding HTTP client
│   └── main.py              # Edge pipeline orchestrator
└── server/
    ├── __init__.py
    ├── app.py               # FastAPI application
    ├── database.py          # SQLite storage
    ├── projector.py         # PyTorch vision projector (40 MB)
    ├── llama_embed.py       # ctypes embedding injection
    ├── inference.py         # Hybrid inference pipeline
    └── main.py              # Server entry point
```
