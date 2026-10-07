import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

spec = importlib.util.spec_from_file_location('device_support', Path(__file__).resolve().parents[1] / 'device_support.py')
support = importlib.util.module_from_spec(spec)
spec.loader.exec_module(support)

class Backend:
    def __init__(self, available=True, count=2):
        self.available, self.count, self.selected = available, count, None
    def is_available(self): return self.available
    def device_count(self): return self.count
    def set_device(self, index): self.selected = index

def device(value):
    kind, _, index = value.partition(':')
    return SimpleNamespace(type=kind, index=int(index) if index else None)

class DeviceTests(unittest.TestCase):
    def setUp(self):
        self.torch = SimpleNamespace(cuda=Backend(), xpu=Backend(), device=device)
    def test_auto_preserves_intel_gpu_index(self):
        self.assertEqual(support.select_device(self.torch, 'auto', device('xpu:1')), ('xpu', 1))
        self.assertEqual(support.initialize_device(self.torch, 'xpu', 1), 'xpu:1')
        self.assertEqual(self.torch.xpu.selected, 1)
        self.assertIsNone(self.torch.cuda.selected)
    def test_explicit_backend_does_not_inherit_another_gpu_index(self):
        self.assertEqual(support.select_device(self.torch, 'cuda', device('xpu:1')), ('cuda', 0))
        self.assertEqual(support.select_device(self.torch, 'cpu', device('xpu:1')), ('cpu', 0))
    def test_external_environment_is_validated_in_worker(self):
        self.torch.xpu.available = False
        self.assertEqual(support.select_device(self.torch, 'xpu', device('cpu'), validate=False), ('xpu', 0))
        with self.assertRaisesRegex(RuntimeError, 'XPU unavailable'):
            support.initialize_device(self.torch, 'xpu', 0)
    def test_missing_backend_or_device_fails_clearly(self):
        self.torch.xpu = None
        with self.assertRaises(RuntimeError): support.select_device(self.torch, 'xpu', device('cpu'))
        with self.assertRaises(ValueError): support.select_device(self.torch, 'mps', device('cpu'))
        with self.assertRaises(RuntimeError): support.select_device(self.torch, 'cuda', device('cuda:3'))

if __name__ == '__main__':
    unittest.main()
