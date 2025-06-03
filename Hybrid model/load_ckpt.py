# File: src/new_hyb_try/load_ckpt.py
# ──────────────────────────────────────────────────────────────────────────────
import torch
import numpy as _np

def safe_load(model, ckpt_path: str, device="cpu"):
    """
    Loads a full checkpoint (state‐dict + extras) even under PyTorch ≥2.6.
    1) Tell torch to allowlist the private NumPy symbol used by older .pth files.
    2) Load with weights_only=False (so we get the entire dict).
    3) If `model` is not None, load model_state (ignoring mismatched keys).
    Returns the full checkpoint dict.
    """
    # 1) allowlist any private NumPy reconstruct call
    torch.serialization.add_safe_globals(
        [ _np.core.multiarray._reconstruct ]
    )

    # 2) load full checkpoint
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # 3) if we gave a model, load the "model_state" field (or the dict itself)
    if model is not None:
        state_dict = ckpt.get("model_state", ckpt)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        loaded = len(state_dict) - len(missing)
        print(f"→ Loaded {loaded} tensors  |  skipped {len(missing)} (size‐mismatch)")
    return ckpt
