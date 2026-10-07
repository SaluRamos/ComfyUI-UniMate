"""Backend selection shared by the ComfyUI adapter and child process."""

def select_device(torch, requested, comfy_device, validate=True):
    kind = comfy_device.type if requested == "auto" else requested
    if kind not in ("cpu", "cuda", "xpu"):
        raise ValueError(f"Backend {kind} unsupported; select cpu, cuda or xpu.")
    index = (comfy_device.index or 0) if comfy_device.type == kind else 0
    if validate and kind != "cpu":
        backend = getattr(torch, kind, None)
        if backend is None or not backend.is_available():
            raise RuntimeError(f"{kind.upper()} unavailable. Install the matching PyTorch build in the selected Python environment.")
        if index >= backend.device_count():
            raise RuntimeError(f"{kind}:{index} is not present in the selected Python environment.")
    return kind, index

def initialize_device(torch, kind, index):
    kind, index = select_device(torch, kind, torch.device(f"{kind}:{index}" if kind != "cpu" else "cpu"))
    if kind != "cpu":
        getattr(torch, kind).set_device(index)
    return f"{kind}:{index}" if kind != "cpu" else "cpu"
