from __future__ import annotations

from pathlib import Path


class UniModelLoadError(RuntimeError):
    pass


def load_uni_model(model_path: Path, device: str) -> torch.nn.Module:
    import torch

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    try:
        model = torch.jit.load(str(model_path), map_location=device)
        model.eval()
        return model
    except Exception:
        pass

    try:
        obj = torch.load(str(model_path), map_location=device)
    except ModuleNotFoundError as exc:
        missing = str(exc)
        raise UniModelLoadError(
            "Model load failed due to missing module dependencies. "
            f"{missing}. If this is a timm-based UNI model, install timm or "
            "provide a TorchScript-exported model."
        ) from exc
    if isinstance(obj, torch.nn.Module):
        obj.eval()
        return obj

    # If the checkpoint is a state_dict, we need a model definition.
    if isinstance(obj, dict) and "state_dict" in obj:
        raise UniModelLoadError(
            "Loaded a state_dict but no model architecture is available. "
            "Provide a TorchScript model or extend the loader with the UNI model definition."
        )

    if isinstance(obj, dict) and "model" in obj and isinstance(obj["model"], torch.nn.Module):
        model = obj["model"]
        model.eval()
        return model

    raise UniModelLoadError(
        "Unable to load UNI model. Provide a TorchScript file or a pickled torch.nn.Module."
    )
