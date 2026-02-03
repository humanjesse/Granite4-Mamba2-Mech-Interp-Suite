"""Load Granite hybrid Mamba-2/Transformer model and inspect architecture."""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# AMD ROCm-specific: override GFX version for GPU compatibility
if torch.cuda.is_available() and hasattr(torch.version, "hip"):
    os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")

DEFAULT_MODEL_ID = "ibm-granite/granite-4.0-h-350m"


def load_model(model_id: str = DEFAULT_MODEL_ID, device: str | None = None):
    """Load model and tokenizer. Returns (model, tokenizer, device)."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Force CPU loading first, then move — avoids fused CUDA kernels
    # that hide intermediates. The slow PyTorch path exposes dt/B/C.
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )

    # Test GPU with a small tensor before committing the whole model
    if device != "cpu":
        try:
            test = torch.zeros(1, device=device)
            del test
            model = model.to(device)
            print(f"Model moved to {device}")
        except Exception as e:
            print(f"GPU move failed ({e}), staying on CPU")
            device = "cpu"

    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer, device


def get_model_info(model):
    """Return dict with model architecture details."""
    config = model.config
    return {
        "model_type": config.model_type,
        "num_layers": config.num_hidden_layers,
        "hidden_size": config.hidden_size,
        "layer_types": list(config.layer_types),
        "mamba_n_heads": config.mamba_n_heads,
        "mamba_d_head": config.mamba_d_head,
        "mamba_d_state": config.mamba_d_state,
        "mamba_n_groups": config.mamba_n_groups,
        "mamba_chunk_size": config.mamba_chunk_size,
        "num_attention_heads": config.num_attention_heads,
        "num_key_value_heads": config.num_key_value_heads,
    }
