from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROGRAM_SYNTHESIS_DIR = os.path.dirname(CURRENT_DIR)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if PROGRAM_SYNTHESIS_DIR not in sys.path:
    sys.path.insert(0, PROGRAM_SYNTHESIS_DIR)

import boosted_runner  # noqa: E402
import posthoc_selector  # noqa: E402


HTRU2_FEATURES = [
    "profile_mean",
    "profile_stdev",
    "profile_skewness",
    "profile_kurtosis",
    "dm_snr_mean",
    "dm_snr_stdev",
    "dm_snr_skewness",
    "dm_snr_kurtosis",
]


@dataclass(frozen=True)
class DistillerParams:
    n_estimators: int
    learning_rate: float
    max_depth: int
    min_samples_leaf: int

    def as_dict(self) -> Dict[str, Any]:
        return {
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "min_samples_leaf": self.min_samples_leaf,
        }


@dataclass
class DistillerCandidate:
    params: DistillerParams
    train_acc: float
    val_acc: float
    test_acc: float
    overfit_gap: float
    stable_val_score: float
    complexity: float
    model: GradientBoostingClassifier

    def to_row(self) -> Dict[str, Any]:
        row = self.params.as_dict()
        row.update(
            {
                "train_acc": self.train_acc,
                "val_acc": self.val_acc,
                "test_acc": self.test_acc,
                "overfit_gap": self.overfit_gap,
                "stable_val_score": self.stable_val_score,
                "complexity": self.complexity,
            }
        )
        return row


def _safe_write_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        delete=False,
        dir=os.path.dirname(path),
    ) as tmp:
        tmp.write(text)
        tmp_path = tmp.name
    os.replace(tmp_path, path)


def _safe_write_json(path: str, obj: Any) -> None:
    _safe_write_text(path, json.dumps(obj, ensure_ascii=False, indent=2))


def write_csv(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        _safe_write_text(path, "")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        newline="",
        delete=False,
        dir=os.path.dirname(path),
    ) as tmp:
        writer = csv.DictWriter(tmp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        tmp_path = tmp.name
    os.replace(tmp_path, path)


def setup_logger(level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("threshold_distiller")
    logger.setLevel(level.upper())
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.handlers[:] = [handler]
    logger.propagate = False
    return logger


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(parsed):
        return default
    return parsed


def infer_numeric_features(
    examples: Sequence[boosted_runner.Example],
    *,
    min_numeric_fraction: float = 0.95,
) -> List[str]:
    if not examples:
        raise ValueError("Cannot infer features from an empty example set.")
    if not isinstance(examples[0].x, dict):
        raise ValueError("Threshold distillation expects tabular dictionary examples.")

    keys: List[str] = []
    for example in examples:
        for key in example.x.keys():
            if key not in keys:
                keys.append(str(key))

    if all(feature in keys for feature in HTRU2_FEATURES):
        return list(HTRU2_FEATURES)

    numeric_features: List[str] = []
    min_count = max(1, int(math.ceil(len(examples) * min_numeric_fraction)))
    for key in keys:
        numeric_count = 0
        for example in examples:
            try:
                value = float(example.x.get(key, ""))
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                numeric_count += 1
        if numeric_count >= min_count:
            numeric_features.append(key)

    if not numeric_features:
        raise ValueError("No sufficiently numeric tabular features were found.")
    return numeric_features


def examples_to_matrix(
    examples: Sequence[boosted_runner.Example],
    features: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray]:
    X = np.array(
        [[_to_float(example.x.get(feature, 0.0)) for feature in features] for example in examples],
        dtype=float,
    )
    y = np.array([example.y01 for example in examples], dtype=int)
    return X, y


def build_param_grid(
    n_estimators: Sequence[int],
    learning_rates: Sequence[float],
    max_depths: Sequence[int],
    min_samples_leaf: Sequence[int],
) -> List[DistillerParams]:
    params: List[DistillerParams] = []
    for n in n_estimators:
        for lr in learning_rates:
            for depth in max_depths:
                for leaf in min_samples_leaf:
                    params.append(
                        DistillerParams(
                            n_estimators=int(n),
                            learning_rate=float(lr),
                            max_depth=int(depth),
                            min_samples_leaf=int(leaf),
                        )
                    )
    return params


def fit_candidates(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    params_grid: Sequence[DistillerParams],
    *,
    overfit_penalty: float,
    random_state: int,
    log: Optional[logging.Logger] = None,
) -> List[DistillerCandidate]:
    candidates: List[DistillerCandidate] = []
    for idx, params in enumerate(params_grid, start=1):
        model = GradientBoostingClassifier(
            n_estimators=params.n_estimators,
            learning_rate=params.learning_rate,
            max_depth=params.max_depth,
            min_samples_leaf=params.min_samples_leaf,
            random_state=random_state,
        )
        model.fit(X_train, y_train)
        train_acc = float(accuracy_score(y_train, model.predict(X_train)))
        val_acc = float(accuracy_score(y_val, model.predict(X_val))) if len(y_val) else 0.0
        test_acc = float(accuracy_score(y_test, model.predict(X_test))) if len(y_test) else 0.0
        overfit_gap = max(0.0, train_acc - val_acc)
        stable_val_score = val_acc - overfit_penalty * overfit_gap
        complexity = (params.n_estimators * params.max_depth) / max(1, params.min_samples_leaf)
        candidates.append(
            DistillerCandidate(
                params=params,
                train_acc=train_acc,
                val_acc=val_acc,
                test_acc=test_acc,
                overfit_gap=overfit_gap,
                stable_val_score=stable_val_score,
                complexity=float(complexity),
                model=model,
            )
        )
        if log and (idx == 1 or idx == len(params_grid) or idx % 100 == 0):
            log.info("fit %d/%d candidates", idx, len(params_grid))
    return candidates


def select_candidate(
    candidates: Sequence[DistillerCandidate],
    *,
    selection_mode: str,
) -> DistillerCandidate:
    if not candidates:
        raise ValueError("Cannot select from an empty candidate list.")
    if selection_mode == "val":
        return max(candidates, key=lambda row: (row.val_acc, -row.complexity, row.stable_val_score))
    if selection_mode == "stable_val":
        return max(candidates, key=lambda row: (row.stable_val_score, row.val_acc, -row.complexity))
    raise ValueError(f"Unknown selection_mode: {selection_mode!r}")


def _tree_node_to_code(
    tree: Any,
    node_id: int,
    *,
    indent: str,
) -> List[str]:
    left = int(tree.children_left[node_id])
    right = int(tree.children_right[node_id])
    if left == right:
        value = float(tree.value[node_id][0][0])
        return [f"{indent}return {value:.17g}"]

    feature_idx = int(tree.feature[node_id])
    threshold = float(tree.threshold[node_id])
    lines = [f"{indent}if values[{feature_idx}] <= {threshold:.17g}:"]
    lines.extend(_tree_node_to_code(tree, left, indent=indent + "    "))
    lines.append(f"{indent}else:")
    lines.extend(_tree_node_to_code(tree, right, indent=indent + "    "))
    return lines


def _base_log_odds(model: GradientBoostingClassifier) -> float:
    prior = getattr(model.init_, "class_prior_", None)
    if prior is None or len(prior) != 2:
        return 0.0
    p0 = max(float(prior[0]), 1e-12)
    p1 = max(float(prior[1]), 1e-12)
    return math.log(p1 / p0)


def build_ensemble_source(
    model: GradientBoostingClassifier,
    *,
    features: Sequence[str],
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    lines: List[str] = [
        "# Auto-generated by threshold_distiller.py",
        "from __future__ import annotations",
        "",
        f"FEATURES = {list(features)!r}",
        f"MODEL_INFO = {metadata or {}}",
        "",
        "def _num(x, key, default=0.0):",
        "    try:",
        "        value = float(x.get(key, default))",
        "    except Exception:",
        "        return default",
        "    if value != value or value in (float('inf'), float('-inf')):",
        "        return default",
        "    return value",
        "",
    ]

    for idx, estimator in enumerate(model.estimators_[:, 0]):
        lines.append(f"def _tree_{idx:03d}(values):")
        lines.extend(_tree_node_to_code(estimator.tree_, 0, indent="    "))
        lines.append("")

    base_score = _base_log_odds(model)
    learning_rate = float(model.learning_rate)
    lines.extend(
        [
            "def predict(x):",
            "    values = [_num(x, feature) for feature in FEATURES]",
            f"    score = {base_score:.17g}",
        ]
    )
    for idx in range(len(model.estimators_[:, 0])):
        lines.append(f"    score += {learning_rate:.17g} * _tree_{idx:03d}(values)")
    lines.extend(
        [
            "    return 1 if score >= 0.0 else 0",
            "",
        ]
    )
    return "\n".join(lines)


def refit_selected(
    selected: DistillerCandidate,
    X_train_val: np.ndarray,
    y_train_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    random_state: int,
) -> Tuple[GradientBoostingClassifier, Dict[str, Any]]:
    params = selected.params
    model = GradientBoostingClassifier(
        n_estimators=params.n_estimators,
        learning_rate=params.learning_rate,
        max_depth=params.max_depth,
        min_samples_leaf=params.min_samples_leaf,
        random_state=random_state,
    )
    model.fit(X_train_val, y_train_val)
    return model, {
        "protocol": "refit_train_val_after_selection",
        "train_val_acc": float(accuracy_score(y_train_val, model.predict(X_train_val))),
        "test_acc": float(accuracy_score(y_test, model.predict(X_test))) if len(y_test) else 0.0,
    }


def run_distillation(args: argparse.Namespace) -> Dict[str, Any]:
    log = setup_logger(args.log_level)
    train_examples, val_examples, test_examples, is_tabular = posthoc_selector.load_dataset_examples(
        fn=args.function,
        length=args.length,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
        dataset_dir=args.dataset_dir,
        tabular_representation=args.tabular_representation,
        cdc_representation=args.cdc_representation,
        log=log,
    )
    if not is_tabular:
        raise ValueError("Threshold distillation currently expects a tabular dataset.")
    if not val_examples:
        raise ValueError("Threshold distillation requires --val-size > 0 for candidate selection.")

    features = args.features or infer_numeric_features(train_examples)
    X_train, y_train = examples_to_matrix(train_examples, features)
    X_val, y_val = examples_to_matrix(val_examples, features)
    X_test, y_test = examples_to_matrix(test_examples, features)

    params_grid = build_param_grid(
        args.n_estimators,
        args.learning_rates,
        args.max_depths,
        args.min_samples_leaf,
    )
    log.info("Fitting %d threshold-ensemble candidates over %d features", len(params_grid), len(features))
    candidates = fit_candidates(
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        params_grid,
        overfit_penalty=args.overfit_penalty,
        random_state=args.random_state,
        log=log,
    )
    selected = select_candidate(candidates, selection_mode=args.selection_mode)
    selected_ranked = sorted(candidates, key=lambda row: (row.stable_val_score, row.val_acc, -row.complexity), reverse=True)
    selected_rank = selected_ranked.index(selected) + 1

    os.makedirs(args.output_dir, exist_ok=True)
    candidate_rows = [candidate.to_row() for candidate in candidates]
    candidate_rows.sort(key=lambda row: (row["stable_val_score"], row["val_acc"], -row["complexity"]), reverse=True)
    write_csv(os.path.join(args.output_dir, "candidate_scores.csv"), candidate_rows)

    metadata = {
        "method": "threshold_distiller",
        "model_family": "sklearn.GradientBoostingClassifier",
        "selection_mode": args.selection_mode,
        "overfit_penalty": args.overfit_penalty,
        "selected_rank_by_stable_val": selected_rank,
        "function": args.function,
        "length": args.length,
        "train_size": args.train_size,
        "val_size": args.val_size,
        "test_size": args.test_size,
        "seed": args.seed,
        "tabular_representation": args.tabular_representation,
        "features": list(features),
        "params": selected.params.as_dict(),
        "train_acc": selected.train_acc,
        "val_acc": selected.val_acc,
        "test_acc": selected.test_acc,
        "overfit_gap": selected.overfit_gap,
        "stable_val_score": selected.stable_val_score,
        "complexity": selected.complexity,
        "candidate_count": len(candidates),
        "baseline_reference_acc": args.baseline_reference_acc,
    }
    _safe_write_json(os.path.join(args.output_dir, "manifest.json"), metadata)
    _safe_write_text(
        os.path.join(args.output_dir, "ensemble.py"),
        build_ensemble_source(selected.model, features=features, metadata=metadata),
    )

    summary_rows: List[Dict[str, Any]] = [
        {
            "protocol": "train_only_selected_by_validation",
            **metadata,
        }
    ]

    refit_metadata: Optional[Dict[str, Any]] = None
    if args.refit_train_val:
        X_train_val = np.vstack([X_train, X_val])
        y_train_val = np.concatenate([y_train, y_val])
        refit_model, refit_scores = refit_selected(
            selected,
            X_train_val,
            y_train_val,
            X_test,
            y_test,
            random_state=args.random_state,
        )
        refit_metadata = {
            **metadata,
            **refit_scores,
            "train_acc": None,
            "val_acc": None,
            "overfit_gap": None,
            "stable_val_score": None,
        }
        _safe_write_json(os.path.join(args.output_dir, "manifest_refit_train_val.json"), refit_metadata)
        _safe_write_text(
            os.path.join(args.output_dir, "ensemble_refit_train_val.py"),
            build_ensemble_source(refit_model, features=features, metadata=refit_metadata),
        )
        summary_rows.append(refit_metadata)

    write_csv(os.path.join(args.output_dir, "summary.csv"), summary_rows)
    result = {
        "selected": metadata,
        "refit": refit_metadata,
        "output_dir": args.output_dir,
    }
    log.info(
        "selected params=%s train=%.4f val=%.4f test=%.4f",
        selected.params.as_dict(),
        selected.train_acc,
        selected.val_acc,
        selected.test_acc,
    )
    if refit_metadata:
        log.info(
            "refit train+val test=%.4f",
            float(refit_metadata["test_acc"]),
        )
    return result


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit a validation-selected local threshold ensemble over named tabular features.",
    )
    parser.add_argument("--function", default="fn_p")
    parser.add_argument("--length", type=int, default=8)
    parser.add_argument("--train-size", type=int, default=256)
    parser.add_argument("--val-size", type=int, default=256)
    parser.add_argument("--test-size", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset-dir", default=boosted_runner.DEFAULT_DATASET_DIR)
    parser.add_argument(
        "--tabular-representation",
        choices=["obfuscated", "semantic", "hybrid", "named_numeric"],
        default="named_numeric",
    )
    parser.add_argument("--cdc-representation", choices=["obfuscated", "semantic"], default=None)
    parser.add_argument("--features", nargs="+", default=None)
    parser.add_argument("--n-estimators", type=int, nargs="+", default=[10, 20, 30, 50, 75, 100, 150, 200])
    parser.add_argument("--learning-rates", type=float, nargs="+", default=[0.02, 0.05, 0.1, 0.2, 0.3])
    parser.add_argument("--max-depths", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--min-samples-leaf", type=int, nargs="+", default=[5, 10, 15, 20, 30, 40])
    parser.add_argument("--selection-mode", choices=["val", "stable_val"], default="stable_val")
    parser.add_argument("--overfit-penalty", type=float, default=1.0)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--baseline-reference-acc", type=float, default=0.9340)
    parser.add_argument("--refit-train-val", action="store_true")
    parser.add_argument("--output-dir", default=os.path.join(CURRENT_DIR, "runs", "htru2_threshold_distiller_named_numeric_s1"))
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    run_distillation(args)


if __name__ == "__main__":
    main()
