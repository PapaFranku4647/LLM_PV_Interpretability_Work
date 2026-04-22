from __future__ import annotations

import argparse
import json
import os
import sys
import logging
from typing import Any, Dict, List, Optional


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(CURRENT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.data_handler import OPENML_BINARY_DATASET_REGISTRY, get_data_generator  # noqa: E402
from src.target_functions import EXPERIMENT_FUNCTION_MAPPING, EXPERIMENT_FUNCTION_METADATA  # noqa: E402


PREFERRED_ORDER = [
    "cardiovascular_disease",
    "heart_disease_comprehensive",
    "breast_wisconsin",
    "wdbc_diagnostic",
    "mammographic_mass",
    "blood_transfusion",
    "chronic_kidney_disease",
    "indian_liver_patient",
]


def reverse_fn_mapping() -> Dict[str, str]:
    return {target: fn for fn, target in EXPERIMENT_FUNCTION_MAPPING.items()}


def recommend_split(pos_count: int, neg_count: int) -> Dict[str, Any]:
    minority = min(pos_count, neg_count)
    result: Dict[str, Any] = {
        "supports_b256": minority >= 128,
        "train_size": None,
        "val_size": None,
        "test_size": None,
        "notes": "",
    }
    if minority < 128:
        result["notes"] = "Minority class below 128; not suitable for balanced batch-256 pilots."
        return result

    train_pc = 128
    remaining_pc = minority - train_pc
    result["train_size"] = 256

    if remaining_pc >= 1128:
        val_pc = 128
        test_pc = 1000
    elif remaining_pc >= 256:
        val_pc = 128
        test_pc = remaining_pc - val_pc
    elif remaining_pc >= 96:
        val_pc = 64
        test_pc = remaining_pc - val_pc
    elif remaining_pc >= 64:
        val_pc = 32
        test_pc = remaining_pc - val_pc
    elif remaining_pc >= 32:
        val_pc = 16
        test_pc = remaining_pc - val_pc
    else:
        val_pc = 0
        test_pc = remaining_pc

    result["val_size"] = 2 * val_pc
    result["test_size"] = 2 * max(0, test_pc)
    if result["val_size"] == 256 and result["test_size"] >= 512:
        result["notes"] = "Good batch-256 pilot candidate."
    elif result["val_size"] >= 128 and result["test_size"] >= 128:
        result["notes"] = "Usable for a batch-256 pilot, but evaluation will be smaller than the main matched CDC setup."
    else:
        result["notes"] = "Supports train batch 256, but validation/test are thin; treat as exploratory only."
    return result


def build_rows() -> List[Dict[str, Any]]:
    fn_lookup = reverse_fn_mapping()
    rows: List[Dict[str, Any]] = []
    for target_name in PREFERRED_ORDER:
        cfg = OPENML_BINARY_DATASET_REGISTRY[target_name]
        fn_id = fn_lookup.get(target_name)
        length = EXPERIMENT_FUNCTION_METADATA[fn_id]["lengths"][0] if fn_id else len(cfg["raw_feature_names"])
        generator = get_data_generator(target_name, length, 2)
        pos_samples, neg_samples = generator._load_dataset()  # internal audit helper
        split = recommend_split(len(pos_samples), len(neg_samples))
        rows.append(
            {
                "target_name": target_name,
                "fn": fn_id,
                "openml_id": cfg["data_id"],
                "features": len(cfg["raw_feature_names"]),
                "positive_count": len(pos_samples),
                "negative_count": len(neg_samples),
                "supports_b256": split["supports_b256"],
                "train_size": split["train_size"],
                "val_size": split["val_size"],
                "test_size": split["test_size"],
                "notes": split["notes"],
                "openml_url": cfg.get("openml_url"),
            }
        )
    return rows


def to_markdown(rows: List[Dict[str, Any]]) -> str:
    headers = [
        "Target",
        "Fn",
        "OpenML",
        "Features",
        "Pos",
        "Neg",
        "b256",
        "Recommended Split",
        "Notes",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        split = (
            f"{row['train_size']}/{row['val_size']}/{row['test_size']}"
            if row["train_size"] is not None
            else "skip"
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    row["target_name"],
                    row["fn"] or "",
                    str(row["openml_id"]),
                    str(row["features"]),
                    str(row["positive_count"]),
                    str(row["negative_count"]),
                    "yes" if row["supports_b256"] else "no",
                    split,
                    row["notes"],
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit and rank CDC-like OpenML pilot datasets.")
    parser.add_argument("--format", choices=["json", "markdown"], default="markdown")
    args = parser.parse_args()

    logging.getLogger("src.data_handler").setLevel(logging.WARNING)
    rows = build_rows()
    if args.format == "json":
        print(json.dumps(rows, indent=2))
    else:
        print(to_markdown(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
