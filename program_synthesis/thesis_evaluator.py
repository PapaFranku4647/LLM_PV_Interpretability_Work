from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Callable, Mapping, Optional, Sequence

try:
    from program_synthesis.live_eval_common import parse_tabular_line, predict_code0_label
except ModuleNotFoundError:
    from live_eval_common import parse_tabular_line, predict_code0_label  # type: ignore


ParseLineFn = Callable[[str], tuple[dict[str, Any], int]]
PredictCode0LabelFn = Callable[[Any, dict[str, Any]], tuple[int, str]]


def load_split_lines(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Split file missing: {path}")
    lines = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and "->" in line:
            lines.append(line)
    return lines


@dataclass(frozen=True)
class ThesisSampleResult:
    x_in_a: bool
    s_size: int
    a_s_size: int
    coverage_ratio: float
    coverage_eq: float
    agreement_count: int
    faithfulness: Optional[float]
    code0_eval_errors: int
    code1_eval_errors: int
    error: Optional[str] = None

    def to_legacy_dict(self) -> dict[str, Any]:
        return {
            "S_size": self.s_size,
            "A_S_size": self.a_s_size,
            "x_in_A": self.x_in_a,
            "coverage_ratio": self.coverage_ratio,
            "coverage_eq": self.coverage_eq,
            "agreement_count": self.agreement_count,
            "faithfulness": self.faithfulness,
            "code0_eval_errors": self.code0_eval_errors,
            "code1_eval_errors": self.code1_eval_errors,
            "error": self.error,
        }


@dataclass(frozen=True)
class ThesisEvaluationReport:
    n_cases: int
    x_in_a_rate: float
    mean_coverage_eq_all: float
    mean_coverage_ratio_all: float
    mean_faithfulness_defined_only: Optional[float]
    mean_faithfulness_all_missing_as_zero: float
    mean_a_s_size: float
    median_a_s_size: float
    code0_eval_error_total: int
    code1_eval_error_total: int

    def to_legacy_dict(self) -> dict[str, Any]:
        return {
            "n_cases": self.n_cases,
            "x_in_A_rate": self.x_in_a_rate,
            "mean_coverage_eq_all": self.mean_coverage_eq_all,
            "mean_coverage_ratio_all": self.mean_coverage_ratio_all,
            "mean_faithfulness_defined_only": self.mean_faithfulness_defined_only,
            "mean_faithfulness_all_missing_as_zero": self.mean_faithfulness_all_missing_as_zero,
            "mean_A_S_size": self.mean_a_s_size,
            "median_A_S_size": self.median_a_s_size,
            "code0_eval_error_total": self.code0_eval_error_total,
            "code1_eval_error_total": self.code1_eval_error_total,
        }


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


class ThesisEvaluator:
    def __init__(
        self,
        code0_fn: Any,
        train_lines: Sequence[str],
        parse_line_fn: Optional[ParseLineFn] = None,
        predict_code0_label_fn: Optional[PredictCode0LabelFn] = None,
    ) -> None:
        self.code0_fn = code0_fn
        self.train_lines = list(train_lines)
        self.parse_line_fn = parse_line_fn or parse_tabular_line
        self.predict_code0_label_fn = predict_code0_label_fn or predict_code0_label

    def evaluate_thesis(
        self,
        sample_x: dict[str, Any],
        pred_label: int,
        check_conditions_fn: Callable[[dict[str, Any]], Any],
    ) -> ThesisSampleResult:
        s_size = len(self.train_lines)
        code1_eval_errors = 0
        code0_eval_errors = 0

        try:
            x_in_a = bool(check_conditions_fn(sample_x))
        except Exception:
            x_in_a = False
            code1_eval_errors += 1

        a_s_size = 0
        agreement_count = 0

        for line in self.train_lines:
            x_i, _ = self.parse_line_fn(line)
            try:
                in_a = bool(check_conditions_fn(x_i))
            except Exception:
                in_a = False
                code1_eval_errors += 1
            if not in_a:
                continue

            a_s_size += 1
            try:
                y_i, _ = self.predict_code0_label_fn(self.code0_fn, x_i)
                if y_i == pred_label:
                    agreement_count += 1
            except Exception:
                code0_eval_errors += 1

        coverage_ratio = (a_s_size / s_size) if s_size > 0 else 0.0
        coverage_eq = coverage_ratio if x_in_a else 0.0
        faithfulness = (agreement_count / a_s_size) if a_s_size > 0 else None
        return ThesisSampleResult(
            x_in_a=x_in_a,
            s_size=s_size,
            a_s_size=a_s_size,
            coverage_ratio=coverage_ratio,
            coverage_eq=coverage_eq,
            agreement_count=agreement_count,
            faithfulness=faithfulness,
            code0_eval_errors=code0_eval_errors,
            code1_eval_errors=code1_eval_errors,
            error=None,
        )

    @staticmethod
    def result_from_mapping(row: Mapping[str, Any]) -> ThesisSampleResult:
        faithfulness = row.get("faithfulness")
        if faithfulness is None:
            faithfulness_f = None
        else:
            faithfulness_f = _as_float(faithfulness, default=0.0)

        return ThesisSampleResult(
            x_in_a=bool(row.get("x_in_A")),
            s_size=_as_int(row.get("S_size"), default=0),
            a_s_size=_as_int(row.get("A_S_size"), default=0),
            coverage_ratio=_as_float(row.get("coverage_ratio"), default=0.0),
            coverage_eq=_as_float(row.get("coverage_eq"), default=0.0),
            agreement_count=_as_int(row.get("agreement_count"), default=0),
            faithfulness=faithfulness_f,
            code0_eval_errors=_as_int(row.get("code0_eval_errors"), default=0),
            code1_eval_errors=_as_int(row.get("code1_eval_errors"), default=0),
            error=row.get("error") if isinstance(row.get("error"), str) else None,
        )

    @staticmethod
    def summarize(results: Sequence[ThesisSampleResult]) -> ThesisEvaluationReport:
        n = len(results)
        if n == 0:
            return ThesisEvaluationReport(
                n_cases=0,
                x_in_a_rate=0.0,
                mean_coverage_eq_all=0.0,
                mean_coverage_ratio_all=0.0,
                mean_faithfulness_defined_only=None,
                mean_faithfulness_all_missing_as_zero=0.0,
                mean_a_s_size=0.0,
                median_a_s_size=0.0,
                code0_eval_error_total=0,
                code1_eval_error_total=0,
            )

        x_in_a_rate = sum(1 for r in results if r.x_in_a) / n
        mean_coverage_eq_all = sum(r.coverage_eq for r in results) / n
        mean_coverage_ratio_all = sum(r.coverage_ratio for r in results) / n

        faithfulness_defined = [r.faithfulness for r in results if r.faithfulness is not None]
        mean_faithfulness_defined_only = (
            (sum(faithfulness_defined) / len(faithfulness_defined))
            if faithfulness_defined
            else None
        )
        mean_faithfulness_all_missing_as_zero = (
            sum((r.faithfulness if r.faithfulness is not None else 0.0) for r in results) / n
        )

        a_s_sizes = [r.a_s_size for r in results]
        mean_a_s_size = sum(a_s_sizes) / n
        median_a_s_size = float(median(a_s_sizes))
        code0_eval_error_total = sum(r.code0_eval_errors for r in results)
        code1_eval_error_total = sum(r.code1_eval_errors for r in results)

        return ThesisEvaluationReport(
            n_cases=n,
            x_in_a_rate=x_in_a_rate,
            mean_coverage_eq_all=mean_coverage_eq_all,
            mean_coverage_ratio_all=mean_coverage_ratio_all,
            mean_faithfulness_defined_only=mean_faithfulness_defined_only,
            mean_faithfulness_all_missing_as_zero=mean_faithfulness_all_missing_as_zero,
            mean_a_s_size=mean_a_s_size,
            median_a_s_size=median_a_s_size,
            code0_eval_error_total=code0_eval_error_total,
            code1_eval_error_total=code1_eval_error_total,
        )
