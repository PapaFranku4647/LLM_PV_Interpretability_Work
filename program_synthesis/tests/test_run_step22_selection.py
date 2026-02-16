from __future__ import annotations

import importlib
import json
import sys
import tempfile
import types
from pathlib import Path
import unittest


PROGRAM_SYNTHESIS_DIR = Path(__file__).resolve().parents[1]
if str(PROGRAM_SYNTHESIS_DIR) not in sys.path:
    sys.path.insert(0, str(PROGRAM_SYNTHESIS_DIR))


def _build_stub_modules() -> dict[str, types.ModuleType]:
    openai_mod = types.ModuleType("openai")

    class DummyOpenAI:
        def __init__(self, *args, **kwargs) -> None:
            self.args = args
            self.kwargs = kwargs

    openai_mod.OpenAI = DummyOpenAI
    return {"openai": openai_mod}


class RunStep22SelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._saved_modules = {}
        for module_name, module_obj in _build_stub_modules().items():
            cls._saved_modules[module_name] = sys.modules.get(module_name)
            sys.modules[module_name] = module_obj

        sys.modules.pop("run_step22_live_once", None)
        cls.step22_mod = importlib.import_module("run_step22_live_once")

    @classmethod
    def tearDownClass(cls) -> None:
        sys.modules.pop("run_step22_live_once", None)
        for module_name, original in cls._saved_modules.items():
            if original is None:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = original

    def test_load_best_row_uses_validation_accuracy(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            trial_path = root / "demo_trial1.jsonl"
            rows = [
                {"attempt": 1, "val_acc": 0.70, "test_acc": 0.98, "code": "def f(x):\n    return 0\n"},
                {"attempt": 2, "val_acc": 0.80, "test_acc": 0.40, "code": "def f(x):\n    return 1\n"},
            ]
            trial_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

            best_file, best_row = self.step22_mod.load_best_row(root)
            self.assertEqual(best_file, "demo_trial1.jsonl")
            self.assertEqual(best_row["attempt"], 2)
            self.assertAlmostEqual(float(best_row["val_acc"]), 0.80)
            self.assertAlmostEqual(float(best_row["test_acc"]), 0.40)

    def test_load_best_row_does_not_use_test_as_tiebreaker(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            trial_path = root / "demo_trial1.jsonl"
            rows = [
                {"attempt": 1, "val_acc": 0.80, "test_acc": 0.20, "code": "def f(x):\n    return 0\n"},
                {"attempt": 2, "val_acc": 0.80, "test_acc": 0.99, "code": "def f(x):\n    return 1\n"},
            ]
            trial_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

            _, best_row = self.step22_mod.load_best_row(root)
            self.assertEqual(best_row["attempt"], 1)
            self.assertAlmostEqual(float(best_row["val_acc"]), 0.80)


if __name__ == "__main__":
    unittest.main()
