# =============================================================================
# Edge Inference VLM - llama.cpp Embedding Injection
# =============================================================================
# Provides functions to inject pre-computed float32 embeddings directly into
# llama.cpp's KV cache via the C-level llama_batch API. This bridges the
# PyTorch projector output with the GGUF quantized LLM.
#
# The standard Python Llama.eval() only accepts token IDs. This module uses
# ctypes to create a llama_batch with the 'embd' field set (float embeddings)
# instead of 'token' (integer IDs), then calls llama_decode directly.
# =============================================================================

import ctypes
import logging
from typing import List

import numpy as np
import llama_cpp

logger = logging.getLogger(__name__)


def eval_embeddings(
    model: "llama_cpp.Llama",
    embeddings: np.ndarray,
    n_past: int,
) -> int:
    """
    Inject float32 embeddings into the llama.cpp KV cache.

    Processes embeddings in chunks that fit within the model's n_batch limit.
    For each chunk, creates a C-level llama_batch with the embd field and
    calls llama_decode to process them through the transformer layers.

    Args:
        model:      The llama_cpp.Llama instance (already loaded).
        embeddings: numpy array of shape (n_tokens, n_embd) with dtype float32.
                    Typically (576, 4096) for LLaVA image features.
        n_past:     Current position in the KV cache (number of tokens
                    already evaluated).

    Returns:
        Updated n_past value (n_past + n_tokens).

    Raises:
        RuntimeError: If llama_decode returns a non-zero error code.
    """
    total_tokens, n_embd = embeddings.shape

    # Ensure contiguous float32 layout for safe ctypes memory copy
    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

    # Process in chunks that fit within llama.cpp's n_batch limit
    batch_size = model.n_batch

    for chunk_start in range(0, total_tokens, batch_size):
        chunk_end = min(chunk_start + batch_size, total_tokens)
        chunk_len = chunk_end - chunk_start
        chunk_data = embeddings[chunk_start:chunk_end]

        # Ensure the chunk is contiguous after slicing
        chunk_data = np.ascontiguousarray(chunk_data, dtype=np.float32)

        # llama_batch_init(n_tokens_alloc, embd, n_seq_max)
        # When embd > 0: allocates float* embd (not llama_token* token)
        batch = llama_cpp.llama_batch_init(chunk_len, n_embd, 1)

        try:
            batch.n_tokens = chunk_len

            # Copy the chunk embeddings into the batch's embd buffer.
            # Layout: flat array of chunk_len * n_embd floats, row-major.
            ctypes.memmove(
                batch.embd,
                chunk_data.ctypes.data,
                chunk_data.nbytes,
            )

            # Set position IDs, sequence IDs, and logits mask
            for i in range(chunk_len):
                batch.pos[i] = n_past + chunk_start + i
                batch.n_seq_id[i] = 1
                batch.seq_id[i][0] = 0
                # Only request logits for the very last token of the very last chunk
                is_last = (chunk_end == total_tokens) and (i == chunk_len - 1)
                batch.logits[i] = 1 if is_last else 0

            # Call the C-level llama_decode with the embedding batch
            rc = llama_cpp.llama_decode(model._ctx.ctx, batch)
            if rc != 0:
                raise RuntimeError(
                    f"llama_decode failed with error code {rc} "
                    f"while injecting chunk [{chunk_start}:{chunk_end}] "
                    f"of {total_tokens} embedding tokens"
                )

            # Update the Llama wrapper's internal token count
            model.n_tokens += chunk_len

        finally:
            # Always free the batch to avoid memory leaks
            llama_cpp.llama_batch_free(batch)

    logger.debug(
        "Injected %d embedding tokens at positions %d..%d (batch_size=%d)",
        total_tokens, n_past, n_past + total_tokens - 1, batch_size,
    )

    return n_past + total_tokens


def generate_with_embeddings(
    model: "llama_cpp.Llama",
    tokens_before: List[int],
    projected_embeddings: np.ndarray,
    tokens_after: List[int],
    max_tokens: int = 256,
    temperature: float = 0.0,
) -> str:
    """
    Generate text from a prompt that includes injected image embeddings.

    Evaluates the prompt in three stages:
        1. Text tokens before the image placeholder
        2. Projected image embeddings (injected via ctypes)
        3. Text tokens after the image placeholder
    Then runs autoregressive generation using llama.cpp's completion API.

    Args:
        model:                The llama_cpp.Llama instance.
        tokens_before:        Token IDs for the text before <image>.
        projected_embeddings: numpy (n_patches, n_embd) float32 from the projector.
        tokens_after:         Token IDs for the text after <image>.
        max_tokens:           Maximum number of tokens to generate.
        temperature:          Sampling temperature (0.0 = greedy).

    Returns:
        The generated text string.
    """
    # Reset the model state (clear KV cache, reset token counter)
    model.reset()
    model._ctx.kv_cache_clear()

    n_past = 0

    # Stage 1: Evaluate text tokens before the image
    if tokens_before:
        model.eval(tokens_before)
        n_past = len(tokens_before)
        logger.debug("Evaluated %d text tokens before image", len(tokens_before))

    # Stage 2: Inject projected image embeddings into the KV cache
    n_past = eval_embeddings(model, projected_embeddings, n_past)
    logger.debug("Injected %d image embedding tokens", projected_embeddings.shape[0])

    # Stage 3: Evaluate text tokens after the image
    if tokens_after:
        model.eval(tokens_after)
        n_past += len(tokens_after)
        logger.debug("Evaluated %d text tokens after image", len(tokens_after))

    # Build the pseudo-prompt from the current input_ids for the completion API.
    # The KV cache already contains the full context; the prompt tokens just
    # need to match what's cached so llama.cpp doesn't re-evaluate them.
    prompt_tokens = model.input_ids[: model.n_tokens].tolist()

    # Run autoregressive generation
    completion = model.create_completion(
        prompt=prompt_tokens,
        max_tokens=max_tokens,
        temperature=temperature,
        stop=["</s>", "USER:"],
    )

    generated_text = completion["choices"][0]["text"].strip()

    logger.debug(
        "Generated %d chars: %s...",
        len(generated_text),
        generated_text[:80],
    )
    return generated_text
