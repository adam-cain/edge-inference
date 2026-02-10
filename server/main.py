# =============================================================================
# Edge Inference VLM - Server Entry Point
# =============================================================================
# CLI entry point for starting the FastAPI server with the hybrid backend:
# PyTorch projector (~33 MB) + llama.cpp GGUF LLM (~4 GB).
# =============================================================================

import argparse
import logging

import uvicorn

from config import get_config


def main():
    """Parse CLI arguments, apply overrides, and start the server."""
    parser = argparse.ArgumentParser(
        description="Edge Inference VLM — Server (hybrid projector + llama.cpp)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", type=str, default=None, help="Server bind address")
    parser.add_argument("--port", type=int, default=None, help="Server bind port")
    parser.add_argument("--model", type=str, default=None, help="Path to GGUF LLM model")
    parser.add_argument("--projector", type=str, default=None, help="Path to projector weights (.pt)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    config = get_config()

    if args.host is not None:
        config.server_host = args.host
    if args.port is not None:
        config.server_port = args.port
    if args.model is not None:
        config.llm_model_path = args.model
    if args.projector is not None:
        config.projector_weights_path = args.projector

    config.server_url = f"http://{config.server_host}:{config.server_port}"

    print("\n" + "=" * 60)
    print("  Edge Inference VLM — Server (hybrid backend)")
    print("=" * 60)
    print(f"  LLM model  : {config.llm_model_path}")
    print(f"  Projector  : {config.projector_weights_path}")
    print(f"  Device     : {config.device}")
    print(f"  GPU layers : {config.n_gpu_layers}")
    print(f"  Database   : {config.db_path}")
    print(f"  Listening  : {config.server_host}:{config.server_port}")
    print("=" * 60 + "\n")

    uvicorn.run(
        "server.app:app",
        host=config.server_host,
        port=config.server_port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
