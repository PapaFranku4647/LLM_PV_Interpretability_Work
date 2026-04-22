from __future__ import annotations

import os
import sys
import tempfile
import unittest


PROGRAM_SYNTHESIS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(PROGRAM_SYNTHESIS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if PROGRAM_SYNTHESIS_DIR not in sys.path:
    sys.path.insert(0, PROGRAM_SYNTHESIS_DIR)

from program_synthesis.boosted import boosted_runner, posthoc_selector  # noqa: E402


def _example(x0: float, y01: int) -> boosted_runner.Example:
    return boosted_runner.Example(
        line=f"x0:{x0} -> {y01}",
        x={"x0": x0},
        y01=y01,
        ypm=1 if y01 == 1 else -1,
    )


class PosthocSelectorTests(unittest.TestCase):
    def test_greedy_selector_can_invert_candidate_predictions(self) -> None:
        train = [_example(0, 0), _example(1, 0), _example(2, 1), _example(3, 1)]
        val = list(train)
        test = list(train)
        inverse_code = "def f(x):\n    return 0 if float(x['x0']) > 1.5 else 1\n"
        weak_code = "def f(x):\n    return 1\n"
        rows = [
            {
                "attempt": 1,
                "round": 1,
                "retry": 1,
                "candidate_code": inverse_code,
                "candidate_code_sha256": boosted_runner._hash_text(inverse_code),
            },
            {
                "attempt": 2,
                "round": 1,
                "retry": 2,
                "candidate_code": weak_code,
                "candidate_code_sha256": boosted_runner._hash_text(weak_code),
            },
        ]

        candidates, rejected = posthoc_selector.load_candidate_programs(rows, train, val, test)
        selected, trace = posthoc_selector.greedy_select_ensemble(
            candidates,
            train,
            val,
            test,
            max_rounds=2,
            max_weak_error=0.499,
            min_alpha=1e-6,
            min_val_improvement=0.0,
            allow_inverted_candidates=True,
        )

        self.assertEqual(rejected, [])
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].candidate.attempt, 1)
        self.assertEqual(selected[0].direction, -1)
        self.assertAlmostEqual(selected[0].val_acc, 1.0)
        self.assertEqual(trace[-1]["stopped_reason"], "no_valid_candidate")

    def test_build_posthoc_ensemble_module_runs_selected_candidate(self) -> None:
        train = [_example(0, 0), _example(1, 0), _example(2, 1), _example(3, 1)]
        code = "def f(x):\n    return 1 if float(x['x0']) > 1.5 else 0\n"
        rows = [
            {
                "attempt": 1,
                "round": 1,
                "retry": 1,
                "candidate_code": code,
                "candidate_code_sha256": boosted_runner._hash_text(code),
            }
        ]
        candidates, _rejected = posthoc_selector.load_candidate_programs(rows, train, train, train)
        selected, _trace = posthoc_selector.greedy_select_ensemble(
            candidates,
            train,
            train,
            train,
            max_rounds=1,
            max_weak_error=0.499,
            min_alpha=1e-6,
            min_val_improvement=0.0,
            allow_inverted_candidates=False,
        )

        module_text = posthoc_selector.build_posthoc_ensemble_module(selected)
        namespace = {}
        exec(compile(module_text, "<posthoc_ensemble>", "exec"), namespace, namespace)

        self.assertEqual(namespace["f"]({"x0": 0}), 0)
        self.assertEqual(namespace["f"]({"x0": 3}), 1)

    def test_uniform_greedy_can_build_complementary_vote_ensemble(self) -> None:
        train = [_example(0, 0), _example(1, 0), _example(2, 1), _example(3, 1)]
        code_a = "def f(x):\n    return 1 if float(x['x0']) > 2.5 else 0\n"
        code_b = "def f(x):\n    return 1 if 1.5 < float(x['x0']) < 2.5 else 0\n"
        rows = [
            {
                "attempt": 1,
                "round": 1,
                "retry": 1,
                "candidate_code": code_a,
                "candidate_code_sha256": boosted_runner._hash_text(code_a),
            },
            {
                "attempt": 2,
                "round": 1,
                "retry": 2,
                "candidate_code": code_b,
                "candidate_code_sha256": boosted_runner._hash_text(code_b),
            },
        ]
        candidates, _rejected = posthoc_selector.load_candidate_programs(rows, train, train, train)

        selected, _trace = posthoc_selector.uniform_greedy_select_ensemble(
            candidates,
            train,
            train,
            train,
            max_rounds=3,
            min_val_improvement=0.0,
            allow_inverted_candidates=False,
        )

        self.assertEqual([item.candidate.attempt for item in selected], [1, 2])
        self.assertAlmostEqual(selected[-1].val_acc, 1.0)

    def test_posthoc_outputs_source_for_selected_candidates(self) -> None:
        train = [_example(0, 0), _example(1, 1)]
        code = "def f(x):\n    return 1 if float(x['x0']) > 0.5 else 0\n"
        selected_candidate = posthoc_selector.SelectedCandidate(
            step=1,
            candidate=posthoc_selector.load_candidate_programs(
                [
                    {
                        "attempt": 1,
                        "round": 1,
                        "retry": 1,
                        "candidate_code": code,
                        "candidate_code_sha256": boosted_runner._hash_text(code),
                    }
                ],
                train,
                train,
                train,
            )[0][0],
            alpha=1.0,
            direction=1,
            weighted_error=0.1,
            train_acc=1.0,
            val_acc=1.0,
            test_acc=1.0,
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            posthoc_selector.write_jsonl(
                os.path.join(tmp_dir, "selected_candidates.jsonl"),
                [
                    {
                        **posthoc_selector.selected_rows([selected_candidate])[0],
                        "candidate_code": selected_candidate.candidate.code,
                    }
                ],
            )
            with open(os.path.join(tmp_dir, "selected_candidates.jsonl"), encoding="utf-8") as handle:
                contents = handle.read()

        self.assertIn("candidate_code", contents)
        self.assertIn("float(x['x0']) > 0.5", contents)


if __name__ == "__main__":
    unittest.main()
