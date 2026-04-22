from __future__ import annotations

import argparse
import base64
import json
import pickle
import statistics
import sys
import textwrap
import zlib
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
from sklearn.tree import export_text


CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import baseline_runner as baseline_runner_mod
import runner as runner_mod
from src.target_functions import EXPERIMENT_FUNCTION_MAPPING


PAPER_PRESETS: Dict[str, Dict[str, Any]] = {
    "cdc_clean": {
        "fn": "fn_o",
        "length": 21,
        "target_name": "cdc_diabetes",
        "train_size": 256,
        "val_size": 256,
        "test_size": 2000,
        "seed": 42,
        "num_trials": 5,
        "tabular_representation": "semantic",
        "tabular_numeric_transform": "positive_affine",
        "codeboost_run_dir": REPO_ROOT / "program_synthesis" / "boosted" / "runs" / "clean_headline_rows_20260421" / "cdc_semantic_schema_affine_s5",
    },
    "pima_clean": {
        "fn": "fn_r",
        "length": 8,
        "target_name": "pima_diabetes",
        "train_size": 256,
        "val_size": 128,
        "test_size": 152,
        "seed": 42,
        "num_trials": 5,
        "tabular_representation": "named_numeric",
        "tabular_numeric_transform": "positive_affine",
        "codeboost_run_dir": REPO_ROOT / "program_synthesis" / "boosted" / "runs" / "clean_headline_rows_20260421" / "pima_named_numeric_none_affine_r40_s5",
    },
    "telco_clean": {
        "fn": "fn_af",
        "length": 19,
        "target_name": "telco_customer_churn",
        "train_size": 256,
        "val_size": 256,
        "test_size": 2000,
        "seed": 42,
        "num_trials": 5,
        "tabular_representation": "named_numeric",
        "tabular_numeric_transform": "positive_affine",
        "codeboost_run_dir": REPO_ROOT / "program_synthesis" / "boosted" / "runs" / "nonmedical_clean_replications_20260421" / "telco_customer_churn_named_numeric_none_affine_s5",
    },
    "credit_g_clean": {
        "fn": "fn_ad",
        "length": 20,
        "target_name": "credit_g",
        "train_size": 256,
        "val_size": 128,
        "test_size": 216,
        "seed": 42,
        "num_trials": 5,
        "tabular_representation": "named_numeric",
        "tabular_numeric_transform": "positive_affine",
        "codeboost_run_dir": REPO_ROOT / "program_synthesis" / "boosted" / "runs" / "nonmedical_clean_replications_20260421" / "credit_g_named_numeric_none_affine_s5",
    },
}


ALL_BASELINE_MODELS = [
    "decision_tree",
    "random_forest",
    "extra_trees",
    "adaboost",
    "gradient_boosting",
    "hist_gradient_boosting",
    "logistic_regression",
    "mlp",
    "xgboost",
]


def _to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, obj: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_to_builtin(dict(obj)), indent=2), encoding="utf-8")


def _feature_bundle(parser: baseline_runner_mod.TabularDataParser) -> Dict[str, Any]:
    return {
        "numeric_features": list(parser.numeric_features),
        "categorical_features": list(parser.categorical_features),
        "category_values": {k: list(v) for k, v in parser.category_values.items()},
        "output_feature_names": list(parser.output_feature_names),
    }


def _exec_wrapper_code(
    model: Any,
    parser: baseline_runner_mod.TabularDataParser,
    *,
    scaled: bool,
    mean: np.ndarray | None,
    std: np.ndarray | None,
) -> str:
    payload = zlib.compress(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL), level=9)
    payload_b64 = base64.b64encode(payload).decode("ascii")
    metadata = {
        "numeric_features": list(parser.numeric_features),
        "categorical_features": list(parser.categorical_features),
        "category_values": {k: list(v) for k, v in parser.category_values.items()},
        "scaled": bool(scaled),
        "mean": mean.tolist() if mean is not None else [],
        "std": std.tolist() if std is not None else [],
    }
    metadata_literal = repr(_to_builtin(metadata))
    return textwrap.dedent(
        f"""
        def f(x, _payload_b64="{payload_b64}", _metadata={metadata_literal}, _cache={{}}):
            import base64
            import pickle
            import zlib

            if not isinstance(x, dict):
                raise TypeError("expected dict sample")

            model = _cache.get("model")
            if model is None:
                model = pickle.loads(zlib.decompress(base64.b64decode(_payload_b64)))
                _cache["model"] = model

            def _to_float(value):
                try:
                    result = float(value)
                except Exception:
                    return 0.0
                if result != result:
                    return 0.0
                if result == float("inf") or result == float("-inf"):
                    return 0.0
                return result

            row = []
            for name in _metadata["numeric_features"]:
                row.append(_to_float(x.get(name, "")))
            for name in _metadata["categorical_features"]:
                value = x.get(name, "?")
                for known in _metadata["category_values"][name]:
                    row.append(1.0 if value == known else 0.0)
            if _metadata["scaled"]:
                scaled_row = []
                for idx, value in enumerate(row):
                    mean = _metadata["mean"][idx]
                    std = _metadata["std"][idx]
                    scaled_row.append((value - mean) / std)
                row = scaled_row

            pred = model.predict([row])[0]
            try:
                pred = pred.item()
            except Exception:
                pass
            if isinstance(pred, bool):
                return 1 if pred else 0
            if isinstance(pred, str):
                pred = pred.strip().strip("'").strip('"')
                if pred in ("0", "1"):
                    return int(pred)
            try:
                return 1 if int(float(pred)) != 0 else 0
            except Exception:
                return 1 if pred else 0
        """
    ).strip() + "\n"


def _dump_tree_text(tree_model: Any, feature_names: Sequence[str]) -> str:
    return export_text(tree_model, feature_names=list(feature_names), decimals=6)


def _dump_hist_tree_nodes(predictor: Any) -> List[Dict[str, Any]]:
    names = list(predictor.nodes.dtype.names or ())
    out: List[Dict[str, Any]] = []
    for raw in predictor.nodes:
        node = {}
        for name in names:
            node[name] = _to_builtin(raw[name])
        out.append(node)
    return out


def build_raw_artifact_text(
    model_name: str,
    model: Any,
    parser: baseline_runner_mod.TabularDataParser,
    *,
    scaled: bool,
    mean: np.ndarray | None,
    std: np.ndarray | None,
    best_params: Mapping[str, Any],
    val_acc: float,
    test_acc: float,
) -> str:
    feature_names = list(parser.output_feature_names)
    header = {
        "model": model_name,
        "best_params": dict(best_params),
        "val_acc": float(val_acc),
        "test_acc": float(test_acc),
        "scaled": bool(scaled),
        "numeric_features": list(parser.numeric_features),
        "categorical_features": list(parser.categorical_features),
        "feature_count_after_encoding": len(feature_names),
    }
    if scaled and mean is not None and std is not None:
        header["scaler_mean"] = mean.tolist()
        header["scaler_std"] = std.tolist()

    if model_name == "decision_tree":
        body = _dump_tree_text(model, feature_names)
        return json.dumps(header, indent=2) + "\n\n" + body + "\n"

    if model_name in {"random_forest", "extra_trees"}:
        sections = [json.dumps({**header, "n_estimators": len(model.estimators_)}, indent=2)]
        for idx, estimator in enumerate(model.estimators_, start=1):
            sections.append(f"=== estimator_{idx} ===")
            sections.append(_dump_tree_text(estimator, feature_names))
        return "\n\n".join(sections) + "\n"

    if model_name == "adaboost":
        sections = [json.dumps({**header, "estimator_weights": _to_builtin(model.estimator_weights_), "estimator_errors": _to_builtin(model.estimator_errors_)}, indent=2)]
        for idx, estimator in enumerate(model.estimators_, start=1):
            sections.append(f"=== estimator_{idx} weight={model.estimator_weights_[idx - 1]} ===")
            sections.append(_dump_tree_text(estimator, feature_names))
        return "\n\n".join(sections) + "\n"

    if model_name == "gradient_boosting":
        sections = [
            json.dumps(
                {
                    **header,
                    "learning_rate": float(model.learning_rate),
                    "init_class": type(model.init_).__name__,
                    "n_estimators": int(model.n_estimators_),
                },
                indent=2,
            )
        ]
        for idx, estimator_arr in enumerate(model.estimators_, start=1):
            estimator = estimator_arr[0]
            sections.append(f"=== stage_{idx} ===")
            sections.append(_dump_tree_text(estimator, feature_names))
        return "\n\n".join(sections) + "\n"

    if model_name == "hist_gradient_boosting":
        predictors = []
        for idx, predictor_arr in enumerate(model._predictors, start=1):
            predictor = predictor_arr[0]
            predictors.append(
                {
                    "stage": idx,
                    "max_depth": int(predictor.get_max_depth()),
                    "n_leaf_nodes": int(predictor.get_n_leaf_nodes()),
                    "nodes": _dump_hist_tree_nodes(predictor),
                }
            )
        payload = {
            **header,
            "learning_rate": float(model.learning_rate),
            "max_iter": int(model.max_iter),
            "n_iter_": int(model.n_iter_),
            "predictors": predictors,
        }
        return json.dumps(payload, indent=2) + "\n"

    if model_name == "logistic_regression":
        payload = {
            **header,
            "classes": _to_builtin(model.classes_),
            "intercept": _to_builtin(model.intercept_),
            "coefficients": [
                {"feature": feature_names[idx], "coefficient": float(value)}
                for idx, value in enumerate(model.coef_[0])
            ],
        }
        return json.dumps(payload, indent=2) + "\n"

    if model_name == "mlp":
        payload = {
            **header,
            "activation": model.activation,
            "out_activation": model.out_activation_,
            "n_layers": int(model.n_layers_),
            "layer_units": [len(layer) if isinstance(layer, np.ndarray) else None for layer in model.coefs_],
            "coefs": _to_builtin(model.coefs_),
            "intercepts": _to_builtin(model.intercepts_),
        }
        return json.dumps(payload, indent=2) + "\n"

    if model_name == "xgboost":
        booster = model.get_booster()
        payload = {
            **header,
            "booster_config": json.loads(booster.save_config()),
            "trees": [json.loads(tree_json) for tree_json in booster.get_dump(dump_format="json")],
        }
        return json.dumps(payload, indent=2) + "\n"

    raise ValueError(f"Unsupported model for artifact export: {model_name}")


def _trial_seed(base_seed: int, trial_index_1based: int) -> int:
    return base_seed + trial_index_1based - 1


def _stable_dataset_seed(fn: str, length: int, train_size: int, val_size: int, test_size: int, base_seed: int) -> int:
    key = f"{fn}|L={length}|train={train_size + val_size}|test={test_size}|base_seed={base_seed}"
    digest = __import__("hashlib").sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFFFFFF


def _baseline_split_paths(cfg: baseline_runner_mod.Config, fn: str, length: int) -> Dict[str, Path]:
    ds = baseline_runner_mod.DatasetStore(cfg)
    derived_seed = ds._stable_derived_seed(fn, length)
    paths = ds._paths(EXPERIMENT_FUNCTION_MAPPING[fn], length, derived_seed)
    ds.get(fn, length)
    return {
        "train": Path(paths["train"]).resolve(),
        "val": Path(paths["val"]).resolve(),
        "test": Path(paths["test"]).resolve(),
        "dir": Path(paths["dir"]).resolve(),
        "dataset_seed": derived_seed,
    }


def _codeboost_split_paths(config: Mapping[str, Any], fn: str, length: int, target_name: str) -> Dict[str, Path | int]:
    dataset_dir = Path(str(config["dataset_dir"]))
    if not dataset_dir.is_absolute():
        dataset_dir = (REPO_ROOT / dataset_dir).resolve()
    derived_seed = _stable_dataset_seed(
        fn=fn,
        length=length,
        train_size=int(config["train_size"]),
        val_size=int(config["val_size"]),
        test_size=int(config["test_size"]),
        base_seed=int(config["seed"]),
    )
    split_dir = dataset_dir / target_name / f"L{length}" / f"seed{derived_seed}"
    return {
        "train": split_dir / "train.txt",
        "val": split_dir / "val.txt",
        "test": split_dir / "test.txt",
        "dir": split_dir,
        "dataset_seed": derived_seed,
    }


def _fit_baseline_trial(
    cfg: baseline_runner_mod.Config,
    fn: str,
    length: int,
    model_name: str,
) -> Iterable[Dict[str, Any]]:
    runner = baseline_runner_mod.BenchmarkRunner(cfg)
    target_name = EXPERIMENT_FUNCTION_MAPPING[fn]
    X_train, y_train, X_val, y_val, X_test, y_test = runner._prepare_data(fn, length, target_name)
    parser = baseline_runner_mod.TabularDataParser(dataset_name=target_name)
    train_lines, val_lines, test_lines = baseline_runner_mod.DatasetStore(cfg).get(fn, length)
    parser.fit_transform(train_lines)
    X_train_scaled, X_val_scaled, X_test_scaled = runner._scaled_arrays(X_train, X_val, X_test)
    param_grids = baseline_runner_mod.get_param_grids(boolean=True)
    split_paths = _baseline_split_paths(cfg, fn, length)

    for trial_zero_idx in range(cfg.num_trials):
        trial_seed = cfg.seed + trial_zero_idx
        use_scaled = model_name in runner.SCALED_MODELS
        Xtr = X_train_scaled if use_scaled else X_train
        Xv = X_val_scaled if use_scaled else X_val
        Xte = X_test_scaled if use_scaled else X_test
        X_select, y_select, selection_split = runner._selection_split(Xtr, y_train, Xv, y_val)
        model, best_params, best_val_acc, _fit_count = runner._fit_select_model(
            model_name,
            Xtr,
            y_train,
            X_select,
            y_select,
            trial_seed,
            param_grids[model_name],
        )
        test_acc = float(baseline_runner_mod.accuracy_score(y_test, model.predict(Xte)))
        mean = X_train.mean(axis=0) if use_scaled else None
        std = X_train.std(axis=0) if use_scaled else None
        if std is not None:
            std = std.copy()
            std[std < 1e-8] = 1.0
        yield {
            "trial_index": trial_zero_idx + 1,
            "trial_seed": trial_seed,
            "selection_split": selection_split,
            "model": model,
            "best_params": best_params,
            "val_acc": float(best_val_acc),
            "test_acc": test_acc,
            "parser": parser,
            "scaled": use_scaled,
            "scale_mean": mean,
            "scale_std": std,
            "split_paths": split_paths,
            "target_name": target_name,
            "train_lines": train_lines,
            "val_lines": val_lines,
            "test_lines": test_lines,
        }


def export_baseline_artifacts(
    *,
    preset: Mapping[str, Any],
    out_root: Path,
    models: Sequence[str],
) -> List[Dict[str, Any]]:
    cfg = baseline_runner_mod.Config(
        functions=[str(preset["fn"])],
        lengths=[int(preset["length"])],
        train_size=int(preset["train_size"]),
        val_size=int(preset["val_size"]),
        test_size=int(preset["test_size"]),
        seed=int(preset["seed"]),
        num_trials=int(preset["num_trials"]),
        tabular_representation=str(preset["tabular_representation"]),
        tabular_numeric_transform=str(preset["tabular_numeric_transform"]),
    )
    if cfg.tabular_representation != "obfuscated":
        cfg.dataset_dir = str(Path(cfg.dataset_dir) / f"tabular_representation_{cfg.tabular_representation}")
    transform_suffix = runner_mod.tabular_numeric_transform_suffix(cfg.tabular_numeric_transform)
    if transform_suffix:
        cfg.dataset_dir = str(Path(cfg.dataset_dir) / transform_suffix)
    exports: List[Dict[str, Any]] = []
    fn = str(preset["fn"])
    length = int(preset["length"])

    for model_name in models:
        method_root = out_root / model_name
        method_root.mkdir(parents=True, exist_ok=True)
        for trial in _fit_baseline_trial(cfg, fn, length, model_name):
            combo_id = f"{fn}_seed{trial['trial_seed']}"
            combo_dir = method_root / combo_id
            combo_dir.mkdir(parents=True, exist_ok=True)
            exec_code = _exec_wrapper_code(
                trial["model"],
                trial["parser"],
                scaled=bool(trial["scaled"]),
                mean=trial["scale_mean"],
                std=trial["scale_std"],
            )
            raw_artifact = build_raw_artifact_text(
                model_name,
                trial["model"],
                trial["parser"],
                scaled=bool(trial["scaled"]),
                mean=trial["scale_mean"],
                std=trial["scale_std"],
                best_params=trial["best_params"],
                val_acc=float(trial["val_acc"]),
                test_acc=float(trial["test_acc"]),
            )
            _write_text(combo_dir / "code_exec.py", exec_code)
            _write_text(combo_dir / "artifact.txt", raw_artifact)
            manifest = {
                "fn": fn,
                "target_name": trial["target_name"],
                "model_name": model_name,
                "length": length,
                "seed": trial["trial_seed"],
                "trial": trial["trial_index"],
                "attempt": 1,
                "dataset_seed": int(trial["split_paths"]["dataset_seed"]),
                "train_path": str(trial["split_paths"]["train"]),
                "val_path": str(trial["split_paths"]["val"]),
                "test_path": str(trial["split_paths"]["test"]),
                "code_path": "code_exec.py",
                "artifact_text_path": "artifact.txt",
                "val_acc": float(trial["val_acc"]),
                "test_acc": float(trial["test_acc"]),
                "selection_split": trial["selection_split"],
                "best_params": trial["best_params"],
                "tabular_representation": cfg.tabular_representation,
                "tabular_numeric_transform": cfg.tabular_numeric_transform,
                "prompt_variant": f"external_raw_{model_name}",
            }
            _write_json(combo_dir / "artifact.json", manifest)
            exports.append(
                {
                    "method": model_name,
                    "combo_id": combo_id,
                    "artifact_dir": str(combo_dir),
                    "val_acc": float(trial["val_acc"]),
                    "test_acc": float(trial["test_acc"]),
                    "artifact_chars": len(raw_artifact),
                }
            )
    return exports


def export_codeboost_artifacts(
    *,
    preset: Mapping[str, Any],
    out_root: Path,
) -> List[Dict[str, Any]]:
    run_dir = Path(preset["codeboost_run_dir"])
    if not run_dir.is_absolute():
        run_dir = (REPO_ROOT / run_dir).resolve()
    trial_dirs = sorted(run_dir.glob(f"{preset['target_name']}/L{preset['length']}/batch*/trial*"))
    exports: List[Dict[str, Any]] = []
    method_root = out_root / "codeboost"
    method_root.mkdir(parents=True, exist_ok=True)

    for trial_dir in trial_dirs:
        manifest_path = trial_dir / "manifest.json"
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        accepted = manifest.get("accepted_rounds") or []
        if not accepted:
            continue
        summary = manifest.get("summary") or {}
        config = manifest.get("config") or {}
        trial_idx = int(summary.get("trial") or trial_dir.name.replace("trial", ""))
        combo_seed = _trial_seed(int(config.get("seed", preset["seed"])), trial_idx)
        combo_id = f"{preset['fn']}_seed{combo_seed}"
        combo_dir = method_root / combo_id
        combo_dir.mkdir(parents=True, exist_ok=True)
        raw_code = str(accepted[-1]["code"])
        _write_text(combo_dir / "code_exec.py", raw_code)
        _write_text(combo_dir / "artifact.txt", raw_code)
        split_paths = _codeboost_split_paths(config, str(preset["fn"]), int(preset["length"]), str(preset["target_name"]))
        artifact_manifest = {
            "fn": str(preset["fn"]),
            "target_name": str(preset["target_name"]),
            "model_name": "codeboost",
            "length": int(preset["length"]),
            "seed": combo_seed,
            "trial": trial_idx,
            "attempt": int(summary.get("api_attempt_count", 0)),
            "dataset_seed": int(split_paths["dataset_seed"]),
            "train_path": str(split_paths["train"]),
            "val_path": str(split_paths["val"]),
            "test_path": str(split_paths["test"]),
            "code_path": "code_exec.py",
            "artifact_text_path": "artifact.txt",
            "val_acc": float(summary.get("final_val_acc")),
            "test_acc": float(summary.get("final_test_acc")),
            "tabular_representation": config.get("tabular_representation"),
            "tabular_numeric_transform": config.get("tabular_numeric_transform"),
            "prompt_variant": "external_raw_codeboost",
        }
        _write_json(combo_dir / "artifact.json", artifact_manifest)
        exports.append(
            {
                "method": "codeboost",
                "combo_id": combo_id,
                "artifact_dir": str(combo_dir),
                "val_acc": float(summary.get("final_val_acc")),
                "test_acc": float(summary.get("final_test_acc")),
                "artifact_chars": len(raw_code),
            }
        )
    return exports


def summarize_exports(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    by_method: Dict[str, List[Mapping[str, Any]]] = {}
    for row in rows:
        by_method.setdefault(str(row["method"]), []).append(row)
    summary = []
    for method, entries in sorted(by_method.items()):
        summary.append(
            {
                "method": method,
                "n_exports": len(entries),
                "mean_val_acc": statistics.mean(float(entry["val_acc"]) for entry in entries),
                "mean_test_acc": statistics.mean(float(entry["test_acc"]) for entry in entries),
                "mean_artifact_chars": statistics.mean(int(entry["artifact_chars"]) for entry in entries),
                "max_artifact_chars": max(int(entry["artifact_chars"]) for entry in entries),
            }
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Export raw textual interpretability artifacts for CodeBoost/baselines.")
    parser.add_argument("--preset", required=True, choices=sorted(PAPER_PRESETS.keys()))
    parser.add_argument("--mode", required=True, choices=["baseline", "codeboost", "both"])
    parser.add_argument("--out-root", default="", help="Output directory for exported artifacts.")
    parser.add_argument("--models", nargs="*", default=list(ALL_BASELINE_MODELS), help="Baseline models to export.")
    args = parser.parse_args()

    preset = PAPER_PRESETS[args.preset]
    out_root = Path(args.out_root) if args.out_root else (REPO_ROOT / "program_synthesis" / "interpretability_artifacts" / args.preset)
    if not out_root.is_absolute():
        out_root = (REPO_ROOT / out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    exported_rows: List[Dict[str, Any]] = []
    if args.mode in {"baseline", "both"}:
        exported_rows.extend(export_baseline_artifacts(preset=preset, out_root=out_root, models=args.models))
    if args.mode in {"codeboost", "both"}:
        exported_rows.extend(export_codeboost_artifacts(preset=preset, out_root=out_root))

    _write_json(out_root / "export_summary.json", {"rows": exported_rows, "summary": summarize_exports(exported_rows)})
    print(json.dumps({"out_root": str(out_root), "summary": summarize_exports(exported_rows)}, indent=2))


if __name__ == "__main__":
    main()
