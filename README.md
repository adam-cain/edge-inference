# Edge Inference VLM

A privacy-preserving vision-language system that captures your screen, encodes it locally through CLIP with AnyRes multi-tile processing, and generates detailed text descriptions via a server-side LLM. **Only abstract embedding vectors leave the device — never raw images.**

## Architecture

```
Edge Client (privacy boundary)                 Server
┌───────────────────────────────┐   HTTP/JSON  ┌─────────────────────────────────┐
│ Screen Capture (mss)          │              │ FastAPI                         │
│        ↓                      │              │   ↓                             │
│ AnyRes Tiling                 │              │ PyTorch Projector (33 MB)       │
│  → overview + grid tiles      │  (N*576,1024)│   ↓                             │
│        ↓                      │ ───────────► │ AnyRes Pack (+ image_newline)   │
│ CLIP ViT-L/14-336             │  float16 b64 │   ↓ (up to 2928, 4096)          │
│ (HuggingFace, MPS)            │              │ ctypes embedding inject         │
│  → encode each tile           │              │   ↓                             │
│        ↓                      │ ◄─────────── │ Mistral-7B GGUF (6 GB Q6_K)    │
│ Privacy: only abstract        │  description │   ↓                             │
│ vectors leave device          │              │ SQLite DB                       │
└───────────────────────────────┘              └─────────────────────────────────┘
```

**Privacy model:** The CLIP vision tower and AnyRes tiling run entirely on the edge device. Only float16 embedding vectors (abstract, non-reconstructable) are transmitted to the server. The server never sees the original screenshots.

**Hybrid backend:** The server uses a lightweight PyTorch projector (~33 MB) to map CLIP embeddings into the LLM's space, packs them with image_newline tile-row delimiters, then injects the packed sequence directly into a GGUF quantized LLM via llama.cpp's C API.

**AnyRes multi-tile:** Instead of compressing the entire screen to 336×336 pixels, the image is divided into an optimal grid of 336×336 tiles (up to 2×2) plus a full-image overview. Each tile is encoded independently through CLIP, giving the LLM ~5× more visual information than a single-crop approach.

## Performance (Apple Silicon M3, 18 GB)

| Metric | Value |
|--------|-------|
| Server memory | ~6.5 GB (GGUF Q6_K + projector) |
| Edge CLIP memory | ~0.6 GB |
| Server load time | ~30 seconds |
| CLIP encode time | ~1.5–2.5 seconds (3–5 tiles) |
| LLM inference | ~8–12 seconds per frame |
| Embedding payload | ~2–6 MB (float16, variable tile count) |
| Image tokens | ~1,752–2,928 (grid-dependent) |

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
# Download the GGUF quantized LLM (~5.9 GB)
python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download('cjpais/llava-1.6-mistral-7b-gguf', 'llava-v1.6-mistral-7b.Q6_K.gguf', local_dir='models')
"

# Extract projector weights + image_newline from HuggingFace LLaVA-NeXT (~33 MB + ~16 KB)
# Downloads the full model once, extracts only the projector components, then frees memory
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

The client loads CLIP (~3s), waits for the server, then begins capturing, tiling, and encoding screenshots.

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
EDGE_MAX_NEW_TOKENS=1024
EDGE_N_CTX=4096
EDGE_CHANGE_DETECTION_THRESHOLD=0.02
EDGE_IMAGE_GRID_PINPOINTS='[[336,672],[672,336],[672,672],[1008,336],[336,1008]]'
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
│   └── extract_projector.py # One-time projector + image_newline extraction
├── models/                  # Model files (downloaded/extracted)
│   ├── llava-v1.6-mistral-7b.Q6_K.gguf  # Mistral-7B GGUF (6 GB)
│   ├── projector_fp16.pt                  # Extracted projector weights (33 MB)
│   ├── image_newline.pt                   # AnyRes tile-row delimiter (16 KB)
│   └── projector_config.json              # Projector architecture + grid config
├── shared/
│   ├── __init__.py
│   └── schemas.py           # Pydantic API schemas (with tile_grid)
├── edge/
│   ├── __init__.py
│   ├── capture.py           # Screen capture (mss)
│   ├── tiling.py            # AnyRes grid selection & image tiling
│   ├── vision.py            # CLIP vision tower (HuggingFace)
│   ├── client.py            # Embedding HTTP client (with tile_grid)
│   └── main.py              # Edge pipeline orchestrator
└── server/
    ├── __init__.py
    ├── app.py               # FastAPI application
    ├── database.py          # SQLite storage
    ├── projector.py         # PyTorch projector + AnyRes tile packing
    ├── llama_embed.py       # ctypes embedding injection
    ├── inference.py         # Hybrid inference pipeline
    └── main.py              # Server entry point
```

## Key Improvements (v0.4.0)

- **AnyRes multi-tile encoding:** Up to 2×2 tile grid gives ~5× more visual tokens
- **LLaVA-NeXT v1.6:** Mistral-7B backbone with stronger instruction following
- **Q6_K quantization:** Near-lossless quality (upgraded from Q4_K)
- **Screen-specific prompt:** Tuned for application names, text, and UI elements
- **Sampling parameters:** Temperature 0.1, repeat penalty 1.1, top-p 0.9
- **Max tokens:** 1,024 (up from 256) — eliminates cut-off descriptions
- **Change detection:** Skips unchanged frames to save inference cycles
- **Sentence-boundary trimming:** Graceful handling if generation hits token limit
