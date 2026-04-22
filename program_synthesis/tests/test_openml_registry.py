from __future__ import annotations

import os
import sys
import unittest


PROGRAM_SYNTHESIS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(PROGRAM_SYNTHESIS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if PROGRAM_SYNTHESIS_DIR not in sys.path:
    sys.path.insert(0, PROGRAM_SYNTHESIS_DIR)

from src.data_handler import OPENML_BINARY_DATASET_REGISTRY, get_data_generator  # noqa: E402
from src.target_functions import EXPERIMENT_FUNCTION_MAPPING, EXPERIMENT_FUNCTION_METADATA  # noqa: E402


class OpenMLRegistryTests(unittest.TestCase):
    def test_registered_targets_have_function_ids_and_lengths(self) -> None:
        reverse_mapping = {target: fn for fn, target in EXPERIMENT_FUNCTION_MAPPING.items()}
        for target_name, cfg in OPENML_BINARY_DATASET_REGISTRY.items():
            self.assertIn(target_name, reverse_mapping)
            fn_id = reverse_mapping[target_name]
            self.assertEqual(EXPERIMENT_FUNCTION_METADATA[fn_id]["lengths"][0], len(cfg["raw_feature_names"]))

    def test_registry_targets_construct_generators(self) -> None:
        for target_name, cfg in OPENML_BINARY_DATASET_REGISTRY.items():
            generator = get_data_generator(target_name, len(cfg["raw_feature_names"]), 2)
            self.assertEqual(generator.DATA_ID, cfg["data_id"])
            self.assertEqual(generator.TARGET_COLUMN, cfg["target_column"])
            self.assertEqual(generator.RAW_FEATURE_NAMES, cfg["raw_feature_names"])


if __name__ == "__main__":
    unittest.main()
