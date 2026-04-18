from __future__ import annotations

import argparse
import logging
import json
import os
import random
import sys
import tempfile
import unittest
from unittest import mock


PROGRAM_SYNTHESIS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(PROGRAM_SYNTHESIS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if PROGRAM_SYNTHESIS_DIR not in sys.path:
    sys.path.insert(0, PROGRAM_SYNTHESIS_DIR)

from program_synthesis import runner as base_runner  # noqa: E402
from program_synthesis.boosted import boosted_runner  # noqa: E402
from src.data_handler import (  # noqa: E402
    CDCDiabetesDataGenerator,
    ChessDataGenerator,
    HTRU2DataGenerator,
    MushroomDataGenerator,
)


class BoostedMathTests(unittest.TestCase):
    def test_estimate_cost_usd_for_gpt_5_2(self) -> None:
        cost = boosted_runner.estimate_cost_usd("gpt-5.2-deep-learning-fundamentals", 1000, 2000)

        self.assertEqual(cost["pricing_model"], "gpt-5.2")
        self.assertAlmostEqual(float(cost["estimated_input_cost_usd"]), 0.00175, places=10)
        self.assertAlmostEqual(float(cost["estimated_output_cost_usd"]), 0.028, places=10)
        self.assertAlmostEqual(float(cost["estimated_total_cost_usd"]), 0.02975, places=10)

    def test_update_distribution_upweights_misclassified_examples(self) -> None:
        weights = [0.25, 0.25, 0.25, 0.25]
        labels_pm = [1, 1, -1, -1]
        preds_pm = [1, -1, -1, 1]

        updated = boosted_runner.update_distribution(weights, labels_pm, preds_pm, alpha=0.5)

        self.assertAlmostEqual(sum(updated), 1.0, places=10)
        self.assertGreater(updated[1], updated[0])
        self.assertGreater(updated[3], updated[2])

    def test_evaluate_weighted_error_counts_failed_predictions_as_errors(self) -> None:
        examples = [
            boosted_runner.Example(line="a -> 0", x="a", y01=0, ypm=-1),
            boosted_runner.Example(line="b -> 1", x="b", y01=1, ypm=1),
            boosted_runner.Example(line="c -> 0", x="c", y01=0, ypm=-1),
            boosted_runner.Example(line="d -> 1", x="d", y01=1, ypm=1),
        ]
        weights = [0.1, 0.2, 0.3, 0.4]

        def fn_callable(x: str) -> int:
            if x == "a":
                return 0
            if x == "b":
                return 0
            if x == "c":
                raise RuntimeError("boom")
            return 1

        weighted_error, preds_pm, eval_errors = boosted_runner.evaluate_weighted_error(
            fn_callable,
            examples,
            weights,
        )

        self.assertAlmostEqual(weighted_error, 0.5, places=10)
        self.assertEqual(preds_pm, [-1, -1, 1, 1])
        self.assertEqual(eval_errors, 1)

    def test_evaluate_ensemble_accuracy_uses_weighted_vote(self) -> None:
        examples = [
            boosted_runner.Example(line="x0:1 -> 1", x={"x0": 1.0}, y01=1, ypm=1),
            boosted_runner.Example(line="x0:-1 -> 0", x={"x0": -1.0}, y01=0, ypm=-1),
        ]
        learners = [
            {"alpha": 0.7, "callable": lambda x: 1 if float(x["x0"]) > 0 else 0},
            {"alpha": 0.2, "callable": lambda _x: 1},
            {"alpha": 0.1, "callable": lambda _x: 0},
        ]

        acc = boosted_runner.evaluate_ensemble_accuracy(learners, examples)

        self.assertAlmostEqual(acc, 1.0, places=10)

    def test_build_ensemble_module_round_trips_predictions(self) -> None:
        learners = [
            {
                "alpha": 0.8,
                "code": "def f_pos(x):\n    return 1 if float(x['x0']) > 0 else 0\n",
            },
            {
                "alpha": 0.3,
                "code": "def f_bias(_x):\n    return 0\n",
            },
        ]

        module_text = boosted_runner.build_ensemble_module(learners)
        namespace: dict[str, object] = {}
        exec(module_text, namespace, namespace)
        ensemble_fn = namespace["f"]

        self.assertEqual(ensemble_fn({"x0": 2.0}), 1)
        self.assertEqual(ensemble_fn({"x0": -2.0}), 0)

    def test_build_repair_prompt_contains_code_mistakes_and_anchors(self) -> None:
        prompt = boosted_runner.build_repair_prompt(
            current_code="def f(x):\n    return 1\n",
            mistake_lines=["x0:-1 -> 0"],
            anchor_lines=["x0:1 -> 1"],
            seq_len=21,
            decimal=False,
            tabular=True,
        )

        self.assertIn("Current code", prompt)
        self.assertIn("def f(x):", prompt)
        self.assertIn("x0:-1 -> 0", prompt)
        self.assertIn("x0:1 -> 1", prompt)
        self.assertIn('{"code": "def f(x):\\n    ..."}', prompt)

    def test_sample_weighted_batch_without_replacement_has_no_duplicates(self) -> None:
        examples = [
            boosted_runner.Example(line=f"x0:{idx} -> {idx % 2}", x={"x0": idx}, y01=idx % 2, ypm=1 if idx % 2 else -1)
            for idx in range(6)
        ]
        weights = [0.05, 0.1, 0.15, 0.2, 0.2, 0.3]

        indices, lines = boosted_runner.sample_weighted_batch(
            examples,
            weights,
            batch_size=6,
            rng=random.Random(7),
            without_replacement=True,
        )

        self.assertEqual(len(indices), 6)
        self.assertEqual(len(set(indices)), 6)
        self.assertEqual(len(lines), 6)

    def test_build_boost_config_strict_acceptance_clamps_thresholds(self) -> None:
        args = argparse.Namespace(
            boost_rounds=8,
            batch_sizes=[256],
            round_retries=3,
            max_weak_error=0.499,
            min_alpha=1e-6,
            num_trials=1,
            stop_on_perfect_train=False,
            resample_each_retry=False,
            output_dir="unused",
            repair_rounds=0,
            repair_mistake_limit=128,
            repair_anchor_count=16,
            repair_trigger_batch_acc_below=None,
            repair_trigger_weighted_error_above=None,
            repair_trigger_min_mistakes=1,
            sample_without_replacement=False,
            strict_acceptance=True,
            whole_train_repair_rounds=0,
            whole_train_repair_batch_size=256,
            whole_train_mistake_frac=0.7,
            whole_train_recent_fix_frac=0.2,
            whole_train_anchor_frac=0.1,
            cdc_representation="obfuscated",
            accept_best_on_failure=False,
            best_fallback_max_weak_error=0.499,
        )

        cfg = boosted_runner.build_boost_config(args)

        self.assertAlmostEqual(cfg.max_weak_error, 0.35, places=10)
        self.assertAlmostEqual(cfg.min_alpha, 0.05, places=10)

    def test_semantic_generation_prompt_includes_cdc_context(self) -> None:
        prompt = boosted_runner.build_generation_prompt(
            ["HighBP:yes,BMI:high,Age:medium -> 1"],
            seq_len=21,
            decimal=False,
            tabular=True,
            dataset_context=boosted_runner.CDC_SEMANTIC_CONTEXT,
        )

        self.assertIn("Dataset: CDC diabetes indicators", prompt)
        self.assertIn("HighBP:yes,BMI:high,Age:medium -> 1", prompt)
        self.assertIn('{"code": "<python function>"}', prompt)

    def test_non_cdc_semantic_contexts_are_selected_by_tabular_representation(self) -> None:
        self.assertIn(
            "secondary mushroom",
            boosted_runner.get_dataset_context("mushroom", "obfuscated", "semantic"),
        )
        self.assertIn(
            "HTRU2 pulsar",
            boosted_runner.get_dataset_context("htru2", "obfuscated", "semantic"),
        )
        self.assertIn(
            "King-Rook",
            boosted_runner.get_dataset_context("chess", "obfuscated", "semantic"),
        )
        self.assertIsNone(boosted_runner.get_dataset_context("mushroom", "obfuscated", "obfuscated"))

    def test_accept_best_on_failure_keeps_best_valid_retry(self) -> None:
        class StubClient:
            def __init__(self) -> None:
                self.cfg = type("Cfg", (), {"seed": 7, "model": "gpt-5"})()

            async def _call_once(self, fn, L, attempt_idx, data_examples, decimal, tabular=False, prompt_override=None):
                if attempt_idx == 1:
                    code = "def f(x):\n    return 1 if float(x['x0']) > -1.5 else 0\n"
                else:
                    code = "def f(x):\n    return 1\n"
                return {
                    "fn": fn,
                    "length": L,
                    "attempt": attempt_idx,
                    "text": json.dumps({"code": code}),
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                    "duration_ms": 1,
                    "tool_uses": 0,
                }

        train_lines = [
            "x0:1 -> 1",
            "x0:2 -> 1",
            "x0:-1 -> 0",
            "x0:-2 -> 0",
        ]
        cfg = boosted_runner.BoostConfig(
            boost_rounds=1,
            batch_sizes=[4],
            round_retries=2,
            max_weak_error=0.1,
            min_alpha=1e-6,
            num_trials=1,
            stop_on_perfect_train=False,
            resample_each_retry=False,
            output_dir="unused",
            accept_best_on_failure=True,
            best_fallback_max_weak_error=0.499,
        )
        log = logging.getLogger("boosted_runner_test")
        log.handlers[:] = [logging.NullHandler()]
        log.propagate = False

        with mock.patch.object(
            boosted_runner,
            "sample_weighted_batch",
            return_value=([0, 1, 2, 3], list(train_lines)),
        ):
            summary, attempt_rows, accepted_rounds = self._run_async(
                boosted_runner.run_boosting_trial(
                    client=StubClient(),
                    log=log,
                    fn="fn_o",
                    length=21,
                    target_name="cdc_diabetes",
                    train_lines=train_lines,
                    val_lines=[],
                    test_lines=list(train_lines),
                    is_decimal=False,
                    is_tabular=True,
                    cfg=cfg,
                    batch_size=4,
                    trial_idx=1,
                )
            )

        self.assertEqual(summary["accepted_rounds"], 1)
        self.assertEqual(summary["stopped_reason"], "max_rounds_reached")
        self.assertAlmostEqual(summary["final_train_acc"], 0.75, places=10)
        self.assertAlmostEqual(summary["final_test_acc"], 0.75, places=10)
        self.assertEqual(len(accepted_rounds), 1)
        self.assertTrue(attempt_rows[0]["accepted"])
        self.assertTrue(attempt_rows[0]["accepted_best_on_failure"])
        self.assertFalse(attempt_rows[1]["accepted"])

    def test_semantic_cdc_generator_uses_raw_feature_names_and_bins(self) -> None:
        def raw_sample(label_shift: int) -> dict[str, str]:
            values = {
                "HighBP": str(label_shift % 2),
                "HighChol": "1",
                "CholCheck": "1",
                "BMI": str(20 + label_shift * 5),
                "Smoker": "0",
                "Stroke": "0",
                "HeartDiseaseorAttack": "0",
                "PhysActivity": "1",
                "Fruits": "1",
                "Veggies": "1",
                "HvyAlcoholConsump": "0",
                "AnyHealthcare": "1",
                "NoDocbcCost": "0",
                "GenHlth": str(1 + label_shift % 5),
                "MentHlth": str(label_shift),
                "PhysHlth": str(label_shift * 2),
                "DiffWalk": "0",
                "Sex": str(label_shift % 2),
                "Age": str(1 + label_shift),
                "Education": str(1 + label_shift % 6),
                "Income": str(1 + label_shift % 8),
            }
            return values

        all_raw = [raw_sample(i) for i in range(10)]
        positive_raw = all_raw[:5]
        negative_raw = all_raw[5:]

        with mock.patch.dict(os.environ, {"CDC_DIABETES_REPRESENTATION": "semantic"}):
            generator = CDCDiabetesDataGenerator(sequence_length=21, num_samples=2)
            generator._load_cached_raw_dataset = mock.Mock(
                return_value=(all_raw, positive_raw, negative_raw)
            )

            positive_samples, negative_samples = generator._load_dataset()

        self.assertIn("HighBP", positive_samples[0])
        self.assertIn("BMI", positive_samples[0])
        self.assertNotIn("x0", positive_samples[0])
        self.assertIn(positive_samples[0]["BMI"], CDCDiabetesDataGenerator.SEMANTIC_BIN_LABELS)
        self.assertIn(negative_samples[0]["Age"], CDCDiabetesDataGenerator.SEMANTIC_BIN_LABELS)
        self.assertIn(positive_samples[0]["Sex"], {"female", "male"})

    def test_semantic_mushroom_generator_uses_named_features_categories_and_bins(self) -> None:
        raw = [
            {
                "cap-diameter": str(1 + idx),
                "cap-shape": "x" if idx % 2 else "f",
                "cap-surface": "s",
                "cap-color": "n",
                "does-bruise-or-bleed": "t" if idx % 2 else "f",
                "gill-attachment": "a",
                "gill-spacing": "c",
                "gill-color": "w",
                "stem-height": str(2 + idx),
                "stem-width": str(3 + idx),
                "stem-root": "b",
                "stem-surface": "s",
                "stem-color": "w",
                "veil-type": "u",
                "veil-color": "w",
                "has-ring": "t",
                "ring-type": "p",
                "spore-print-color": "k",
                "habitat": "d",
                "season": "u",
            }
            for idx in range(10)
        ]

        with mock.patch.dict(os.environ, {"TABULAR_REPRESENTATION": "semantic"}):
            generator = MushroomDataGenerator(sequence_length=20, num_samples=2)
            generator._init_semantic_bins(raw)
            sample = generator._semantic_sample(raw[0])

        self.assertIn("cap_diameter", sample)
        self.assertNotIn("x0", sample)
        self.assertIn(sample["cap_diameter"], ["very low", "low", "medium", "high", "very high"])
        self.assertEqual(sample["cap_shape"], "flat")
        self.assertEqual(sample["has_ring"], "yes")
        self.assertEqual(sample["habitat"], "woods")

    def test_semantic_htru2_generator_uses_named_binned_features(self) -> None:
        raw = [
            {name: str(idx + feat_idx) for feat_idx, name in enumerate(HTRU2DataGenerator.RAW_FEATURE_NAMES)}
            for idx in range(10)
        ]

        with mock.patch.dict(os.environ, {"TABULAR_REPRESENTATION": "semantic"}):
            generator = HTRU2DataGenerator(sequence_length=8, num_samples=2)
            generator._init_semantic_bins(raw)
            sample = generator._semantic_sample(raw[0])

        self.assertIn("profile_mean", sample)
        self.assertIn("dm_snr_kurtosis", sample)
        self.assertNotIn("x0", sample)
        self.assertIn(sample["profile_mean"], ["very low", "low", "medium", "high", "very high"])

    def test_semantic_chess_generator_uses_uci_names_and_readable_values(self) -> None:
        raw = {name: "t" for name in ChessDataGenerator.RAW_FEATURE_NAMES}
        raw["dsopp"] = "g"
        raw["hdchk"] = "w"
        raw["wkpos"] = "n"

        with mock.patch.dict(os.environ, {"TABULAR_REPRESENTATION": "semantic"}):
            generator = ChessDataGenerator(sequence_length=35, num_samples=2)
            sample = generator._semantic_sample(raw)

        self.assertEqual(sample["bkblk"], "true")
        self.assertEqual(sample["dsopp"], "greater")
        self.assertEqual(sample["hdchk"], "white")
        self.assertEqual(sample["wkpos"], "none")

    def test_run_boosting_trial_stops_on_perfect_train(self) -> None:
        class StubClient:
            def __init__(self) -> None:
                self.cfg = type("Cfg", (), {"seed": 7})()

            async def _call_once(self, fn, L, attempt_idx, data_examples, decimal, tabular=False):
                return {
                    "fn": fn,
                    "length": L,
                    "attempt": attempt_idx,
                    "text": "{\"code\":\"def f(x):\\n    return 1 if float(x['x0']) > 0 else 0\\n\"}",
                    "duration_ms": 1,
                    "tool_uses": 0,
                }

        train_lines = [
            "x0:1 -> 1",
            "x0:2 -> 1",
            "x0:-1 -> 0",
            "x0:-2 -> 0",
        ]
        test_lines = list(train_lines)
        cfg = boosted_runner.BoostConfig(
            boost_rounds=3,
            batch_sizes=[4],
            round_retries=1,
            max_weak_error=0.499,
            min_alpha=1e-6,
            num_trials=1,
            stop_on_perfect_train=True,
            resample_each_retry=False,
            output_dir="unused",
        )
        log = logging.getLogger("boosted_runner_test")
        log.handlers[:] = [logging.NullHandler()]
        log.propagate = False

        summary, attempt_rows, accepted_rounds = self._run_async(
            boosted_runner.run_boosting_trial(
                client=StubClient(),
                log=log,
                fn="fn_o",
                length=21,
                target_name="cdc_diabetes",
                train_lines=train_lines,
                val_lines=[],
                test_lines=test_lines,
                is_decimal=False,
                is_tabular=True,
                cfg=cfg,
                batch_size=4,
                trial_idx=1,
            )
        )

        self.assertEqual(summary["accepted_rounds"], 1)
        self.assertEqual(summary["stopped_reason"], "perfect_train_acc")
        self.assertAlmostEqual(summary["final_train_acc"], 1.0, places=10)
        self.assertAlmostEqual(summary["final_test_acc"], 1.0, places=10)
        self.assertEqual(len(accepted_rounds), 1)
        self.assertTrue(attempt_rows[0]["accepted"])

    def test_repair_loop_can_turn_failed_initial_candidate_into_accepted_learner(self) -> None:
        class StubClient:
            def __init__(self) -> None:
                self.cfg = type("Cfg", (), {"seed": 7, "model": "gpt-5"})()
                self.prompt_overrides = []

            async def _call_once(
                self,
                fn,
                L,
                attempt_idx,
                data_examples,
                decimal,
                tabular=False,
                prompt_override=None,
            ):
                self.prompt_overrides.append(prompt_override)
                if prompt_override is None:
                    code = "def f(x):\n    return 1\n"
                else:
                    code = "def f(x):\n    return 1 if float(x['x0']) > 0 else 0\n"
                return {
                    "fn": fn,
                    "length": L,
                    "attempt": attempt_idx,
                    "text": "{\"code\":\"" + code.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"') + "\"}",
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                    "duration_ms": 1,
                    "tool_uses": 0,
                }

        train_lines = [
            "x0:1 -> 1",
            "x0:2 -> 1",
            "x0:-1 -> 0",
            "x0:-2 -> 0",
        ]
        cfg = boosted_runner.BoostConfig(
            boost_rounds=1,
            batch_sizes=[4],
            round_retries=1,
            max_weak_error=0.499,
            min_alpha=1e-6,
            num_trials=1,
            stop_on_perfect_train=True,
            resample_each_retry=False,
            output_dir="unused",
            repair_rounds=1,
            repair_mistake_limit=10,
            repair_anchor_count=2,
        )
        log = logging.getLogger("boosted_runner_test")
        log.handlers[:] = [logging.NullHandler()]
        log.propagate = False
        client = StubClient()

        with mock.patch.object(
            boosted_runner,
            "sample_weighted_batch",
            return_value=([0, 1, 2, 3], list(train_lines)),
        ):
            summary, attempt_rows, accepted_rounds = self._run_async(
                boosted_runner.run_boosting_trial(
                    client=client,
                    log=log,
                    fn="fn_o",
                    length=21,
                    target_name="cdc_diabetes",
                    train_lines=train_lines,
                    val_lines=[],
                    test_lines=list(train_lines),
                    is_decimal=False,
                    is_tabular=True,
                    cfg=cfg,
                    batch_size=4,
                    trial_idx=1,
                )
            )

        self.assertEqual(summary["api_attempt_count"], 2)
        self.assertEqual(summary["accepted_rounds"], 1)
        self.assertEqual(len(accepted_rounds), 1)
        self.assertEqual(len(client.prompt_overrides), 2)
        self.assertIsNone(client.prompt_overrides[0])
        self.assertIn("Wrong examples", client.prompt_overrides[1])
        self.assertTrue(attempt_rows[0]["accepted"])
        self.assertEqual(attempt_rows[0]["repair_calls"], 1)
        self.assertAlmostEqual(attempt_rows[0]["repair_initial_batch_acc"], 0.5, places=10)
        self.assertAlmostEqual(attempt_rows[0]["repair_final_batch_acc"], 1.0, places=10)
        self.assertAlmostEqual(summary["final_train_acc"], 1.0, places=10)

    def test_repair_loop_falls_back_after_no_code_response(self) -> None:
        class StubClient:
            def __init__(self) -> None:
                self.cfg = type("Cfg", (), {"seed": 7, "model": "gpt-5"})()
                self.prompt_overrides = []

            async def _call_once(
                self,
                fn,
                L,
                attempt_idx,
                data_examples,
                decimal,
                tabular=False,
                prompt_override=None,
            ):
                self.prompt_overrides.append(prompt_override)
                if prompt_override is None:
                    text = json.dumps({"code": "def f(x):\n    return 1\n"})
                elif "Return only valid JSON" in prompt_override:
                    text = json.dumps({"code": "def f(x):\n    return 1 if float(x['x0']) > 0 else 0\n"})
                else:
                    text = "analysis without code"
                return {
                    "fn": fn,
                    "length": L,
                    "attempt": attempt_idx,
                    "text": text,
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                    "duration_ms": 1,
                    "tool_uses": 0,
                }

        train_lines = [
            "x0:1 -> 1",
            "x0:2 -> 1",
            "x0:-1 -> 0",
            "x0:-2 -> 0",
        ]
        cfg = boosted_runner.BoostConfig(
            boost_rounds=1,
            batch_sizes=[4],
            round_retries=1,
            max_weak_error=0.499,
            min_alpha=1e-6,
            num_trials=1,
            stop_on_perfect_train=True,
            resample_each_retry=False,
            output_dir="unused",
            repair_rounds=1,
            repair_mistake_limit=10,
            repair_anchor_count=2,
        )
        log = logging.getLogger("boosted_runner_test")
        log.handlers[:] = [logging.NullHandler()]
        log.propagate = False
        client = StubClient()

        with mock.patch.object(
            boosted_runner,
            "sample_weighted_batch",
            return_value=([0, 1, 2, 3], list(train_lines)),
        ):
            summary, attempt_rows, _accepted_rounds = self._run_async(
                boosted_runner.run_boosting_trial(
                    client=client,
                    log=log,
                    fn="fn_o",
                    length=21,
                    target_name="cdc_diabetes",
                    train_lines=train_lines,
                    val_lines=[],
                    test_lines=list(train_lines),
                    is_decimal=False,
                    is_tabular=True,
                    cfg=cfg,
                    batch_size=4,
                    trial_idx=1,
                )
            )

        history = json.loads(attempt_rows[0]["repair_history"])
        self.assertEqual(summary["api_attempt_count"], 3)
        self.assertEqual(attempt_rows[0]["repair_calls"], 2)
        self.assertEqual(history[1]["status"], "no_code_found")
        self.assertEqual(history[2]["status"], "fallback_repair")
        self.assertAlmostEqual(attempt_rows[0]["repair_final_batch_acc"], 1.0, places=10)

    def test_adaptive_repair_gate_skips_repair_for_good_initial_candidate(self) -> None:
        class StubClient:
            def __init__(self) -> None:
                self.cfg = type("Cfg", (), {"seed": 7, "model": "gpt-5"})()
                self.prompt_overrides = []

            async def _call_once(
                self,
                fn,
                L,
                attempt_idx,
                data_examples,
                decimal,
                tabular=False,
                prompt_override=None,
            ):
                self.prompt_overrides.append(prompt_override)
                return {
                    "fn": fn,
                    "length": L,
                    "attempt": attempt_idx,
                    "text": json.dumps({"code": "def f(x):\n    return 1 if float(x['x0']) > 0 else 0\n"}),
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                    "duration_ms": 1,
                    "tool_uses": 0,
                }

        train_lines = [
            "x0:1 -> 1",
            "x0:2 -> 1",
            "x0:-1 -> 0",
            "x0:-2 -> 0",
        ]
        cfg = boosted_runner.BoostConfig(
            boost_rounds=1,
            batch_sizes=[4],
            round_retries=1,
            max_weak_error=0.499,
            min_alpha=1e-6,
            num_trials=1,
            stop_on_perfect_train=True,
            resample_each_retry=False,
            output_dir="unused",
            repair_rounds=1,
            repair_mistake_limit=10,
            repair_anchor_count=2,
            repair_trigger_batch_acc_below=0.75,
        )
        log = logging.getLogger("boosted_runner_test")
        log.handlers[:] = [logging.NullHandler()]
        log.propagate = False
        client = StubClient()

        with mock.patch.object(
            boosted_runner,
            "sample_weighted_batch",
            return_value=([0, 1, 2, 3], list(train_lines)),
        ):
            summary, attempt_rows, accepted_rounds = self._run_async(
                boosted_runner.run_boosting_trial(
                    client=client,
                    log=log,
                    fn="fn_o",
                    length=21,
                    target_name="cdc_diabetes",
                    train_lines=train_lines,
                    val_lines=[],
                    test_lines=list(train_lines),
                    is_decimal=False,
                    is_tabular=True,
                    cfg=cfg,
                    batch_size=4,
                    trial_idx=1,
                )
            )

        self.assertEqual(summary["api_attempt_count"], 1)
        self.assertEqual(summary["accepted_rounds"], 1)
        self.assertEqual(len(accepted_rounds), 1)
        self.assertEqual(attempt_rows[0]["repair_calls"], 0)
        self.assertEqual(attempt_rows[0]["repair_gate_reason"], "skipped_gate_min_mistakes")
        self.assertEqual(len(client.prompt_overrides), 1)

    def test_whole_train_repair_can_improve_train_candidate(self) -> None:
        class StubClient:
            def __init__(self) -> None:
                self.cfg = type("Cfg", (), {"seed": 7, "model": "gpt-5"})()
                self.prompt_overrides = []

            async def _call_once(
                self,
                fn,
                L,
                attempt_idx,
                data_examples,
                decimal,
                tabular=False,
                prompt_override=None,
            ):
                self.prompt_overrides.append(prompt_override)
                if prompt_override is None:
                    code = "def f(x):\n    return 1 if float(x['x0']) > 1 else 0\n"
                else:
                    code = "def f(x):\n    return 1 if float(x['x0']) > 0 else 0\n"
                return {
                    "fn": fn,
                    "length": L,
                    "attempt": attempt_idx,
                    "text": json.dumps({"code": code}),
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                    "duration_ms": 1,
                    "tool_uses": 0,
                }

        train_lines = [
            "x0:1 -> 1",
            "x0:2 -> 1",
            "x0:-1 -> 0",
            "x0:-2 -> 0",
        ]
        cfg = boosted_runner.BoostConfig(
            boost_rounds=1,
            batch_sizes=[4],
            round_retries=1,
            max_weak_error=0.499,
            min_alpha=1e-6,
            num_trials=1,
            stop_on_perfect_train=False,
            resample_each_retry=False,
            output_dir="unused",
            repair_rounds=0,
            whole_train_repair_rounds=1,
            whole_train_repair_batch_size=4,
            whole_train_mistake_frac=0.7,
            whole_train_recent_fix_frac=0.2,
            whole_train_anchor_frac=0.1,
        )
        log = logging.getLogger("boosted_runner_test")
        log.handlers[:] = [logging.NullHandler()]
        log.propagate = False
        client = StubClient()

        with mock.patch.object(
            boosted_runner,
            "sample_weighted_batch",
            return_value=([0, 1, 2, 3], list(train_lines)),
        ):
            summary, attempt_rows, accepted_rounds = self._run_async(
                boosted_runner.run_boosting_trial(
                    client=client,
                    log=log,
                    fn="fn_o",
                    length=21,
                    target_name="cdc_diabetes",
                    train_lines=train_lines,
                    val_lines=[],
                    test_lines=list(train_lines),
                    is_decimal=False,
                    is_tabular=True,
                    cfg=cfg,
                    batch_size=4,
                    trial_idx=1,
                )
            )

        history = json.loads(attempt_rows[0]["repair_history"])
        self.assertEqual(summary["api_attempt_count"], 2)
        self.assertEqual(summary["accepted_rounds"], 1)
        self.assertEqual(len(accepted_rounds), 1)
        self.assertAlmostEqual(summary["final_train_acc"], 1.0, places=10)
        self.assertAlmostEqual(summary["final_test_acc"], 1.0, places=10)
        self.assertEqual(attempt_rows[0]["repair_calls"], 1)
        self.assertIn("iterative curriculum repair loop", client.prompt_overrides[1])
        self.assertEqual(history[-1]["status"], "whole_train_repair")
        self.assertTrue(history[-1]["accepted_as_best"])

    def test_semantic_cdc_trial_uses_context_prompt_and_named_features(self) -> None:
        class StubClient:
            def __init__(self) -> None:
                self.cfg = type("Cfg", (), {"seed": 7, "model": "gpt-5"})()
                self.prompt_overrides = []

            async def _call_once(
                self,
                fn,
                L,
                attempt_idx,
                data_examples,
                decimal,
                tabular=False,
                prompt_override=None,
            ):
                self.prompt_overrides.append(prompt_override)
                code = (
                    "def f(x):\n"
                    "    return 1 if x.get('HighBP') == 'yes' or x.get('BMI') in ('high', 'very high') else 0\n"
                )
                return {
                    "fn": fn,
                    "length": L,
                    "attempt": attempt_idx,
                    "text": json.dumps({"code": code}),
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                    "duration_ms": 1,
                    "tool_uses": 0,
                }

        train_lines = [
            "HighBP:yes,BMI:medium,Age:high -> 1",
            "HighBP:no,BMI:very high,Age:medium -> 1",
            "HighBP:no,BMI:low,Age:low -> 0",
            "HighBP:no,BMI:medium,Age:very low -> 0",
        ]
        cfg = boosted_runner.BoostConfig(
            boost_rounds=1,
            batch_sizes=[4],
            round_retries=1,
            max_weak_error=0.499,
            min_alpha=1e-6,
            num_trials=1,
            stop_on_perfect_train=True,
            resample_each_retry=False,
            output_dir="unused",
            cdc_representation="semantic",
        )
        log = logging.getLogger("boosted_runner_test")
        log.handlers[:] = [logging.NullHandler()]
        log.propagate = False
        client = StubClient()

        with mock.patch.object(
            boosted_runner,
            "sample_weighted_batch",
            return_value=([0, 1, 2, 3], list(train_lines)),
        ):
            summary, attempt_rows, accepted_rounds = self._run_async(
                boosted_runner.run_boosting_trial(
                    client=client,
                    log=log,
                    fn="fn_o",
                    length=21,
                    target_name="cdc_diabetes",
                    train_lines=train_lines,
                    val_lines=[],
                    test_lines=list(train_lines),
                    is_decimal=False,
                    is_tabular=True,
                    cfg=cfg,
                    batch_size=4,
                    trial_idx=1,
                )
            )

        self.assertEqual(summary["accepted_rounds"], 1)
        self.assertAlmostEqual(summary["final_train_acc"], 1.0, places=10)
        self.assertAlmostEqual(summary["final_test_acc"], 1.0, places=10)
        self.assertEqual(len(accepted_rounds), 1)
        self.assertTrue(attempt_rows[0]["accepted"])
        self.assertIsNotNone(client.prompt_overrides[0])
        self.assertIn("Dataset: CDC diabetes indicators", client.prompt_overrides[0])
        self.assertIn("HighBP:yes,BMI:medium,Age:high -> 1", client.prompt_overrides[0])

    @staticmethod
    def _run_async(awaitable):
        import asyncio

        return asyncio.run(awaitable)


class TamuClientConfigTests(unittest.TestCase):
    def test_load_dotenv_file_sets_missing_values_without_overriding_existing_env(self) -> None:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as handle:
            handle.write("FOO=from_file\n")
            handle.write("BAR=\"quoted\"\n")
            dotenv_path = handle.name
        try:
            with mock.patch.dict(os.environ, {"FOO": "from_env"}, clear=False):
                loaded = base_runner._load_dotenv_file(dotenv_path)
                self.assertTrue(loaded)
                self.assertEqual(os.environ["FOO"], "from_env")
                self.assertEqual(os.environ["BAR"], "quoted")
        finally:
            os.unlink(dotenv_path)

    def test_config_reads_tamu_env_aliases(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "TAMU_API_KEY": "tamu-key",
                "TAMU_AZURE_ENDPOINT": "https://example.openai.azure.com/",
                "TAMU_API_VERSION": "2024-12-01-preview",
                "API_BASE_URL": "https://chat-api.tamu.ai/openai",
                "API_MODE": "chat_completions",
            },
            clear=True,
        ):
            cfg = base_runner.Config()

        self.assertEqual(cfg.api_key, "tamu-key")
        self.assertEqual(cfg.api_base_url, "https://chat-api.tamu.ai/openai")
        self.assertEqual(cfg.azure_endpoint, "https://example.openai.azure.com/")
        self.assertEqual(cfg.api_version, "2024-12-01-preview")
        self.assertEqual(cfg.api_mode, "chat_completions")

    def test_create_openai_client_selects_azure_when_endpoint_present(self) -> None:
        class DummyAzureClient:
            def __init__(self, **kwargs) -> None:
                self.kwargs = kwargs

        class DummyOpenAIClient:
            def __init__(self, **kwargs) -> None:
                self.kwargs = kwargs

        cfg = base_runner.Config()
        cfg.api_key = "key"
        cfg.azure_endpoint = "https://example.openai.azure.com/"
        cfg.api_version = "2024-12-01-preview"
        cfg.api_base_url = ""

        with mock.patch.object(base_runner, "AsyncAzureOpenAI", DummyAzureClient), mock.patch.object(
            base_runner, "AsyncOpenAI", DummyOpenAIClient
        ):
            client, client_type = base_runner.create_openai_client(cfg)

        self.assertIsInstance(client, DummyAzureClient)
        self.assertEqual(client_type, "azure_openai")
        self.assertEqual(client.kwargs["azure_endpoint"], cfg.azure_endpoint.rstrip("/"))
        self.assertEqual(client.kwargs["api_version"], cfg.api_version)

    def test_create_openai_client_uses_base_url_for_openai_compatible_hosts(self) -> None:
        class DummyOpenAIClient:
            def __init__(self, **kwargs) -> None:
                self.kwargs = kwargs

        cfg = base_runner.Config()
        cfg.api_key = "key"
        cfg.api_base_url = "https://example.internal/v1"
        cfg.azure_endpoint = ""
        cfg.api_version = ""

        with mock.patch.object(base_runner, "AsyncOpenAI", DummyOpenAIClient):
            client, client_type = base_runner.create_openai_client(cfg)

        self.assertIsInstance(client, DummyOpenAIClient)
        self.assertEqual(client_type, "openai_compatible")
        self.assertEqual(client.kwargs["base_url"], cfg.api_base_url)

    def test_create_openai_client_prefers_azure_key_when_endpoint_present(self) -> None:
        class DummyAzureClient:
            def __init__(self, **kwargs) -> None:
                self.kwargs = kwargs

        cfg = base_runner.Config()
        cfg.api_key = "openai-like-key"
        cfg.azure_endpoint = "https://example.openai.azure.com/"
        cfg.api_version = "2024-12-01-preview"
        cfg.api_key_explicit = False

        with mock.patch.dict(
            os.environ,
            {"AZURE_OPENAI_API_KEY": "azure-key"},
            clear=False,
        ), mock.patch.object(base_runner, "AsyncAzureOpenAI", DummyAzureClient):
            client, client_type = base_runner.create_openai_client(cfg)

        self.assertIsInstance(client, DummyAzureClient)
        self.assertEqual(client_type, "azure_openai")
        self.assertEqual(client.kwargs["api_key"], "azure-key")

    def test_resolve_openai_request_model_maps_known_azure_alias(self) -> None:
        cfg = base_runner.Config()
        cfg.model = "protected.gpt-5.2"
        cfg.azure_endpoint = "https://example.openai.azure.com/"

        resolved = base_runner.resolve_openai_request_model(cfg)

        self.assertEqual(resolved, "gpt-5.2-deep-learning-fundamentals")

    def test_resolve_openai_request_model_prefers_explicit_deployment_env(self) -> None:
        cfg = base_runner.Config()
        cfg.model = "protected.gpt-5.2"
        cfg.azure_endpoint = "https://example.openai.azure.com/"

        with mock.patch.dict(
            os.environ,
            {"TAMU_DEPLOYMENT": "custom-deployment"},
            clear=False,
        ):
            resolved = base_runner.resolve_openai_request_model(cfg)

        self.assertEqual(resolved, "custom-deployment")


if __name__ == "__main__":
    unittest.main()
