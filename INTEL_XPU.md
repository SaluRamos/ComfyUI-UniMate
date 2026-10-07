# Intel Arc / Arc Pro B70

The adapter accepts `device=xpu`; `auto` follows ComfyUI's selected GPU. CUDA and CPU remain available. Explicit GPU indices are preserved in the child process, including T5 AMP. A separate `python_executable` is validated inside that interpreter.

Install the current Intel graphics/compute driver for your OS, then install the native XPU build in the Python environment used for inference:

```sh
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu
python -m pip install -r requirements.txt
python -c "import torch; assert torch.xpu.is_available(); print(torch.xpu.get_device_name(0))"
```

The existing Motion dependency still needs its build prerequisites described in README.md. Restart ComfyUI and select `auto` on an XPU ComfyUI installation, or select `xpu` explicitly. Start with batch_size=1. The child uses native PyTorch operations; CUDA kernels are not required by this adapter.

The repository previously omitted vendor/unimate/models and vendor/unimate/dataset because its ignore patterns also matched those source directories. They are restored from the upstream revision recorded in THIRD_PARTY.md, and model/data cache ignore rules are now rooted at the repository root.

Validation performed: device dispatch tests with mocked CPU/CUDA/XPU backends, including multiple devices, external environments and unavailable hardware; Python compilation. Actual model inference and third-party dependency behavior on Arc Pro B70 were not measured in the development environment, which lacks PyTorch and an Intel GPU. This is implemented XPU support awaiting hardware validation, not a B70 benchmark or certification.

Run `python -m unittest discover -s tests -p test_devices.py -v`, then exercise a short text-to-motion workflow with the checkpoint and feature cache available. Verify output animation before increasing batch size.
