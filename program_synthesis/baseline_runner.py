"""Baseline ML models for comparison with LLM-ERM."""

import os, sys, json, csv, time, argparse, random, hashlib, pickle
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Optional
import logging
import numpy as np
from itertools import product

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from sklearn.svm import SVC
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover - optional dependency
    XGBClassifier = None

try:
    from tabpfn import TabPFNClassifier
except Exception:  # pragma: no cover - optional dependency
    TabPFNClassifier = None

from src.data_handler import get_data_generator, create_stratified_splits
from src.target_functions import EXPERIMENT_FUNCTION_MAPPING, EXPERIMENT_FUNCTION_METADATA

FUNCTION_NAME_MAPPING = EXPERIMENT_FUNCTION_MAPPING
TABULAR_FNS = {"adult_income", "mushroom", "cdc_diabetes", "htru2", "chess"}
BOOLEAN_FNS = {"parity_all", "parity_first_half", "parity_rand_3", "parity_rand_10", 
               "automata_parity", "palindrome", "dyck2", "patternmatch1", "patternmatch2",
               "prime_decimal", "prime_decimal_tf_check", "sha256_parity", "prime_plus_47", "collatz_steps_parity",
               "graph_has_cycle", "graph_connected",
               "adult_income", "mushroom", "cdc_diabetes", "htru2", "chess"}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("baseline_runner")

OPS_UNARY = ["NOT"]
OPS_BINARY = ["XOR", "AND", "OR"]
OPS_ALL = OPS_UNARY + OPS_BINARY


@dataclass
class Node:
    op: str
    idx: Optional[int] = None
    val: Optional[int] = None
    left: Optional["Node"] = None
    right: Optional["Node"] = None
    
    def is_leaf(self) -> bool:
        return self.op in ("VAR", "CONST")
    
    def size(self) -> int:
        if self.is_leaf():
            return 1
        if self.op in OPS_UNARY:
            return 1 + self.left.size()
        return 1 + self.left.size() + self.right.size()
    
    def depth(self) -> int:
        """Returns the maximum depth of the tree (number of edges from root to deepest leaf)."""
        if self.is_leaf():
            return 0
        if self.op in OPS_UNARY:
            return 1 + self.left.depth()
        return 1 + max(self.left.depth(), self.right.depth())
    
    def clone(self) -> "Node":
        if self.is_leaf():
            return Node(self.op, self.idx, self.val)
        if self.op in OPS_UNARY:
            return Node(self.op, left=self.left.clone())
        return Node(self.op, left=self.left.clone(), right=self.right.clone())
    
    def eval(self, X: np.ndarray) -> np.ndarray:
        if self.op == "VAR":
            return X[:, self.idx].astype(bool)
        if self.op == "CONST":
            return np.full(X.shape[0], bool(self.val), dtype=bool)
        if self.op == "NOT":
            return np.logical_not(self.left.eval(X))
        a = self.left.eval(X)
        b = self.right.eval(X)
        if self.op == "XOR":
            return np.logical_xor(a, b)
        if self.op == "AND":
            return np.logical_and(a, b)
        if self.op == "OR":
            return np.logical_or(a, b)
        raise ValueError(f"Unknown op: {self.op}")
    
    def to_str(self) -> str:
        if self.op == "VAR":
            return f"x{self.idx}"
        if self.op == "CONST":
            return str(self.val)
        if self.op == "NOT":
            return f"(~{self.left.to_str()})"
        sym = {"XOR": "^", "AND": "&", "OR": "|"}[self.op]
        return f"({self.left.to_str()} {sym} {self.right.to_str()})"


def random_leaf(d: int, p_const: float = 0.1) -> Node:
    if random.random() < p_const:
        return Node("CONST", val=random.randint(0, 1))
    return Node("VAR", idx=random.randrange(d))


def random_tree(d: int, max_depth: int, p_const: float = 0.1) -> Node:
    if max_depth <= 0 or random.random() < 0.25:
        return random_leaf(d, p_const=p_const)
    op = random.choice(OPS_ALL)
    if op in OPS_UNARY:
        return Node(op, left=random_tree(d, max_depth - 1, p_const=p_const))
    return Node(
        op,
        left=random_tree(d, max_depth - 1, p_const=p_const),
        right=random_tree(d, max_depth - 1, p_const=p_const),
    )


def iter_nodes_with_parents(root: Node):
    stack = [(root, None, None)]
    while stack:
        node, parent, is_left = stack.pop()
        yield node, parent, is_left
        if node.is_leaf():
            continue
        if node.op in OPS_UNARY:
            stack.append((node.left, node, True))
        else:
            stack.append((node.right, node, False))
            stack.append((node.left, node, True))


def pick_random_subtree(root: Node) -> Tuple[Node, Optional[Node], Optional[bool]]:
    nodes = list(iter_nodes_with_parents(root))
    return random.choice(nodes)


def replace_child(parent: Node, is_left: bool, new_child: Node):
    if parent.op in OPS_UNARY:
        parent.left = new_child
    else:
        if is_left:
            parent.left = new_child
        else:
            parent.right = new_child


def crossover(a: Node, b: Node) -> Tuple[Node, Node]:
    a2 = a.clone()
    b2 = b.clone()
    na, pa, is_left_a = pick_random_subtree(a2)
    nb, pb, is_left_b = pick_random_subtree(b2)
    if pa is None and pb is None:
        return b2, a2
    if pa is None:
        a2 = nb.clone()
    else:
        replace_child(pa, is_left_a, nb.clone())
    if pb is None:
        b2 = na.clone()
    else:
        replace_child(pb, is_left_b, na.clone())
    return a2, b2


def mutate(root: Node, d: int, max_depth: int, p_const: float = 0.1) -> Node:
    r = root.clone()
    n, p, is_left = pick_random_subtree(r)
    new_sub = random_tree(d, max_depth=max_depth, p_const=p_const)
    if p is None:
        return new_sub
    replace_child(p, is_left, new_sub)
    return r


def ga_accuracy(pred_bool: np.ndarray, y: np.ndarray) -> float:
    yb = y.astype(bool)
    return float(np.mean(pred_bool == yb))


class GeneticAlgorithmClassifier:
    def __init__(
        self,
        pop_size: int = 300,
        generations: int = 80,
        max_depth_init: int = 4,
        max_depth_mut: int = 4,
        tournament_k: int = 5,
        cx_prob: float = 0.6,
        mut_prob: float = 0.3,
        elite_frac: float = 0.05,
        size_penalty: float = 1e-4,
        fitness_subsample: int = 4000,
        random_state: int = 0,
    ):
        self.pop_size = pop_size
        self.generations = generations
        self.max_depth_init = max_depth_init
        self.max_depth_mut = max_depth_mut
        self.tournament_k = tournament_k
        self.cx_prob = cx_prob
        self.mut_prob = mut_prob
        self.elite_frac = elite_frac
        self.size_penalty = size_penalty
        self.fitness_subsample = fitness_subsample
        self.random_state = random_state
        self.best_node_ = None
        self.logs_ = None
    
    def fit(self, X: np.ndarray, y: np.ndarray, X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None):
        random.seed(self.random_state)
        np.random.seed(self.random_state)
        
        X = X.astype(bool)
        y = y.astype(int)
        n, d = X.shape
        
        def eval_fitness(node: Node) -> float:
            if self.fitness_subsample is not None and self.fitness_subsample < n:
                idx = np.random.choice(n, size=self.fitness_subsample, replace=False)
                pred = node.eval(X[idx])
                acc = ga_accuracy(pred, y[idx])
            else:
                pred = node.eval(X)
                acc = ga_accuracy(pred, y)
            return acc - self.size_penalty * node.size()
        
        def tournament_select(pop: List[Node], fits: List[float]) -> Node:
            cand = random.sample(range(len(pop)), k=min(self.tournament_k, len(pop)))
            best_i = max(cand, key=lambda i: fits[i])
            return pop[best_i]
        
        pop = [random_tree(d, self.max_depth_init) for _ in range(self.pop_size)]
        logs = {"best_train_acc": [], "best_val_acc": [], "best_size": []}
        best_node = None
        best_fit = -1e9
        
        for gen in range(self.generations):
            fits = [eval_fitness(ind) for ind in pop]
            gen_best_i = int(np.argmax(fits))
            gen_best = pop[gen_best_i]
            gen_best_fit = fits[gen_best_i]
            
            if gen_best_fit > best_fit:
                best_fit = gen_best_fit
                best_node = gen_best.clone()
            
            train_acc = ga_accuracy(best_node.eval(X), y)
            if X_val is not None and y_val is not None:
                X_val_b = X_val.astype(bool)
                y_val_i = y_val.astype(int)
                val_acc = ga_accuracy(best_node.eval(X_val_b), y_val_i)
            else:
                val_acc = float("nan")
            
            logs["best_train_acc"].append(train_acc)
            logs["best_val_acc"].append(val_acc)
            logs["best_size"].append(best_node.size())
            
            elite_n = max(1, int(self.elite_frac * self.pop_size))
            elite_idx = np.argsort(fits)[-elite_n:]
            new_pop = [pop[i].clone() for i in elite_idx]
            
            while len(new_pop) < self.pop_size:
                r = random.random()
                if r < self.cx_prob and len(new_pop) + 1 < self.pop_size:
                    p1 = tournament_select(pop, fits)
                    p2 = tournament_select(pop, fits)
                    c1, c2 = crossover(p1, p2)
                    new_pop.extend([c1, c2])
                elif r < self.cx_prob + self.mut_prob:
                    p = tournament_select(pop, fits)
                    c = mutate(p, d=d, max_depth=self.max_depth_mut)
                    new_pop.append(c)
                else:
                    p = tournament_select(pop, fits)
                    new_pop.append(p.clone())
            
            pop = new_pop[:self.pop_size]
        
        self.best_node_ = best_node
        self.logs_ = logs
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.best_node_ is None:
            raise RuntimeError("Model not fitted yet")
        X_bool = X.astype(bool)
        return self.best_node_.eval(X_bool).astype(int)


@dataclass
class Config:
    functions: List[str] = field(default_factory=lambda: [
        "fn_m", "fn_n", "fn_o", "fn_p", "fn_q",
    ])
    lengths: List[int] = field(default_factory=lambda: [100, 50, 30, 25, 20])
    train_size: int = int(os.getenv("TRAIN_SIZE", "100"))
    val_size: int = int(os.getenv("VAL_SIZE", "100"))
    test_size: int = int(os.getenv("TEST_SIZE", "10000"))
    seed: int = int(os.getenv("GLOBAL_SEED", "42"))
    num_trials: int = int(os.getenv("NUM_TRIALS", "10"))
    dataset_dir: str = os.getenv("DATASET_DIR", "program_synthesis/datasets")
    tabular_representation: str = os.getenv("TABULAR_REPRESENTATION", "obfuscated")
    out_jsonl: str = "program_synthesis/baseline_results.jsonl"
    out_csv: str = "program_synthesis/baseline_results.csv"
    models: List[str] = field(default_factory=lambda: [
        "decision_tree",
        "random_forest",
        "extra_trees",
        "adaboost",
        "gradient_boosting",
        "hist_gradient_boosting",
        "logistic_regression",
        "mlp",
        "xgboost",
    ])
    include_ga: bool = False
    include_tabpfn: bool = False
    max_train_rows: Optional[int] = None
    max_test_rows: Optional[int] = None
    save_model_artifacts: bool = False


class DatasetStore:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    @staticmethod
    def _set_seed(seed: int):
        random.seed(seed)
        np.random.seed(seed)
        try:
            import torch
            torch.manual_seed(seed)
        except Exception:
            pass

    def _paths(self, target_name: str, L: int, derived_seed: int) -> Dict[str, str]:
        base = os.path.join(self.cfg.dataset_dir, target_name, f"L{L}", f"seed{derived_seed}")
        return {
            "dir": base,
            "train": os.path.join(base, "train.txt"),
            "val": os.path.join(base, "val.txt"),
            "test": os.path.join(base, "test.txt"),
        }

    def _stable_derived_seed(self, fn: str, L: int) -> int:
        key = f"{fn}|L={L}|train={self.cfg.train_size+self.cfg.val_size}|test={self.cfg.test_size}|base_seed={self.cfg.seed}"
        digest = hashlib.sha256(key.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big") & 0x7FFFFFFF

    def _read_lines(self, path: str) -> List[str]:
        with open(path, "r", encoding="utf-8") as f:
            return [ln.rstrip("\n") for ln in f]

    def get(self, fn: str, L: int) -> Tuple[List[str], List[str], List[str]]:
        target_name = FUNCTION_NAME_MAPPING[fn]
        os.environ["TABULAR_REPRESENTATION"] = self.cfg.tabular_representation
        os.environ["CDC_DIABETES_REPRESENTATION"] = self.cfg.tabular_representation
        os.environ["MUSHROOM_REPRESENTATION"] = self.cfg.tabular_representation
        os.environ["HTRU2_REPRESENTATION"] = self.cfg.tabular_representation
        os.environ["CHESS_REPRESENTATION"] = self.cfg.tabular_representation
        derived_seed = self._stable_derived_seed(fn, L)
        paths = self._paths(target_name, L, derived_seed)

        if all(os.path.exists(paths[k]) for k in ["train", "val", "test"]):
            return self._read_lines(paths["train"]), self._read_lines(paths["val"]), self._read_lines(paths["test"])

        self._set_seed(derived_seed)
        total_samples = self.cfg.train_size + self.cfg.val_size + self.cfg.test_size
        generator = get_data_generator(target_name, L, total_samples)
        all_samples = generator.generate_data()

        train_split, val_split, test_split = create_stratified_splits(
            all_samples, self.cfg.train_size, self.cfg.val_size, self.cfg.test_size, device='cpu'
        )

        os.makedirs(paths["dir"], exist_ok=True)
        for split_name, split_data in [("train", train_split), ("val", val_split), ("test", test_split)]:
            with open(paths[split_name], "w") as f:
                for s in split_data:
                    input_val = s['Input']
                    if isinstance(input_val, np.ndarray):
                        input_str = input_val.item() if input_val.size == 1 else ''.join(input_val.tolist())
                    else:
                        input_str = ''.join(input_val) if isinstance(input_val, (list, tuple)) else str(input_val)
                    f.write(f"{input_str} -> {s['Output']}\n")

        return self._read_lines(paths["train"]), self._read_lines(paths["val"]), self._read_lines(paths["test"])


def parse_data(lines: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for line in lines:
        seq, label = line.split("->")
        # Encode characters: digits 0-9 stay as-is, 'u' -> 10, 'v' -> 11, filter out spaces/others
        encoded = []
        for c in seq.strip():
            if c.isdigit():
                encoded.append(int(c))
            elif c == 'u':
                encoded.append(10)
            elif c == 'v':
                encoded.append(11)
            # Skip spaces and other characters
        X.append(encoded)
        y.append(int(label.strip()))
    return np.array(X), np.array(y)


class TabularDataParser:
    """Dense tabular encoder for generated `key:value,... -> y` rows.

    Numeric columns are inferred from train values. Every other column is one-hot
    encoded from train categories. This supports both obfuscated rows (`x3:1.2`)
    and semantic rows (`HighBP:yes,BMI:high`) without imposing ordinal structure
    on categorical values.
    """

    def __init__(self, dataset_name: str = 'adult_income'):
        self.dataset_name = dataset_name
        self.feature_names: List[str] = []
        self._fitted = False
        self.numeric_features: List[str] = []
        self.categorical_features: List[str] = []
        self.numeric_indices: List[int] = []
        self.categorical_indices: List[int] = []
        self.category_values: Dict[str, List[str]] = {}
        self.output_feature_names: List[str] = []

    def _parse_line(self, line: str) -> Tuple[Dict[str, str], int]:
        features_str, label = line.split("->")
        features: Dict[str, str] = {}
        for pair in features_str.strip().split(","):
            if ":" not in pair:
                continue
            key, value = pair.split(":", 1)
            features[key.strip()] = value.strip()
        return features, int(label.strip())

    @staticmethod
    def _as_float(value: str) -> Optional[float]:
        try:
            result = float(value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(result):
            return None
        return result

    def _infer_columns(self, parsed: List[Tuple[Dict[str, str], int]]) -> None:
        seen: List[str] = []
        for features, _label in parsed:
            for name in features:
                if name not in seen:
                    seen.append(name)
        self.feature_names = seen

        self.numeric_features = []
        self.categorical_features = []
        for name in self.feature_names:
            values = [features.get(name, "") for features, _label in parsed]
            non_missing = [value for value in values if value not in {"", "?", "nan", "None"}]
            if non_missing and all(self._as_float(value) is not None for value in non_missing):
                self.numeric_features.append(name)
            else:
                self.categorical_features.append(name)

        self.numeric_indices = [self.feature_names.index(name) for name in self.numeric_features]
        self.categorical_indices = [self.feature_names.index(name) for name in self.categorical_features]
        self.category_values = {
            name: sorted({features.get(name, "?") for features, _label in parsed})
            for name in self.categorical_features
        }
        self.output_feature_names = list(self.numeric_features)
        for name in self.categorical_features:
            self.output_feature_names.extend(f"{name}={value}" for value in self.category_values[name])

    def fit_transform(self, lines: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        parsed = [self._parse_line(line) for line in lines]
        if not parsed:
            return np.zeros((0, 0), dtype=float), np.array([], dtype=int)
        self._infer_columns(parsed)
        self._fitted = True
        return self._transform(parsed)

    def transform(self, lines: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        if not self._fitted:
            raise RuntimeError("Parser must be fitted before transform")
        parsed = [self._parse_line(line) for line in lines]
        return self._transform(parsed)

    def _transform(self, parsed: List[Tuple[Dict[str, str], int]]) -> Tuple[np.ndarray, np.ndarray]:
        rows: List[List[float]] = []
        labels: List[int] = []
        for features, label in parsed:
            row: List[float] = []
            for name in self.numeric_features:
                row.append(self._as_float(features.get(name, "")) or 0.0)
            for name in self.categorical_features:
                value = features.get(name, "?")
                row.extend(1.0 if value == known else 0.0 for known in self.category_values[name])
            rows.append(row)
            labels.append(label)
        return np.asarray(rows, dtype=float), np.asarray(labels, dtype=int)


class GABinarizer:
    def __init__(self):
        self.thresholds = None
    
    def fit_transform(
        self,
        X: np.ndarray,
        numeric_indices: Optional[List[int]] = None,
        categorical_indices: Optional[List[int]] = None,
    ) -> np.ndarray:
        del numeric_indices, categorical_indices
        if X.size == 0:
            self.thresholds = np.array([])
            return np.zeros((X.shape[0], 0), dtype=int)
        self.thresholds = np.median(X, axis=0)
        return (X > self.thresholds).astype(int)
    
    def transform(
        self,
        X: np.ndarray,
        numeric_indices: Optional[List[int]] = None,
        categorical_indices: Optional[List[int]] = None,
    ) -> np.ndarray:
        del numeric_indices, categorical_indices
        if self.thresholds is None:
            raise RuntimeError("Binarizer must be fitted before transform")
        if X.size == 0:
            return np.zeros((X.shape[0], 0), dtype=int)
        return (X > self.thresholds).astype(int)


def get_param_grids(boolean: bool = False) -> Dict[str, Dict[str, List[Any]]]:
    """Small validation-selection grids for reproducible benchmark runs."""
    grids: Dict[str, Dict[str, List[Any]]] = {
        "decision_tree": {
            "max_depth": [3, 5, 8, 12, None],
            "min_samples_leaf": [1, 5, 20],
            "criterion": ["gini", "entropy"],
        },
        "random_forest": {
            "n_estimators": [128, 256],
            "max_depth": [8, 16, None],
            "min_samples_leaf": [1, 5],
        },
        "extra_trees": {
            "n_estimators": [128, 256],
            "max_depth": [8, 16, None],
            "min_samples_leaf": [1, 5],
        },
        "adaboost": {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.05, 0.1, 0.5, 1.0],
        },
        "gradient_boosting": {
            "n_estimators": [100, 200],
            "learning_rate": [0.03, 0.1],
            "max_depth": [2, 3],
        },
        "hist_gradient_boosting": {
            "max_iter": [100, 200],
            "learning_rate": [0.03, 0.1],
            "max_leaf_nodes": [15, 31],
        },
        "logistic_regression": {
            "C": [0.1, 1.0, 10.0],
        },
        "mlp": {
            "hidden_layer_sizes": [(64,), (128,), (64, 32)],
            "alpha": [1e-4, 1e-3],
        },
        "svm": {
            "C": [0.5, 1.0, 5.0],
            "gamma": ["scale"],
            "kernel": ["rbf"],
        },
        "xgboost": {
            "n_estimators": [100, 200],
            "max_depth": [3, 5],
            "learning_rate": [0.05, 0.1],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
        },
        "tabpfn": {},
    }

    if boolean:
        grids["genetic_algorithm"] = {
            "pop_size": [300],
            "generations": [80],
            "max_depth_init": [6, 10],
            "max_depth_mut": [4],
            "cx_prob": [0.7],
            "mut_prob": [0.25],
        }
    return grids


def get_base_models(seed: int, include_ga: bool = False, include_tabpfn: bool = False) -> Dict[str, Any]:
    models: Dict[str, Any] = {
        "decision_tree": DecisionTreeClassifier(random_state=seed),
        "random_forest": RandomForestClassifier(random_state=seed, n_jobs=-1),
        "extra_trees": ExtraTreesClassifier(random_state=seed, n_jobs=-1),
        "adaboost": AdaBoostClassifier(random_state=seed),
        "gradient_boosting": GradientBoostingClassifier(random_state=seed),
        "hist_gradient_boosting": HistGradientBoostingClassifier(random_state=seed),
        "logistic_regression": LogisticRegression(max_iter=2000, random_state=seed),
        "mlp": MLPClassifier(max_iter=500, early_stopping=True, random_state=seed),
        "svm": SVC(random_state=seed),
    }

    if XGBClassifier is not None:
        models["xgboost"] = XGBClassifier(
            booster="gbtree",
            random_state=seed,
            objective="binary:logistic",
            eval_metric="logloss",
            verbosity=0,
            tree_method="hist",
            device="cpu",
            n_jobs=1,
        )

    if include_ga:
        models["genetic_algorithm"] = GeneticAlgorithmClassifier(random_state=seed)

    if include_tabpfn and TabPFNClassifier is not None:
        models["tabpfn"] = TabPFNClassifier()

    return models


def iter_param_grid(param_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    if not param_grid:
        return [{}]
    keys = list(param_grid.keys())
    return [dict(zip(keys, values)) for values in product(*(param_grid[key] for key in keys))]


class BaselineRunner:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.ds = DatasetStore(cfg)
        self.best_decision_trees = {}
        self.best_decision_tree_accs = {}
        self.best_decision_tree_infos = {}
        self.best_ga_models = {}
        self.best_ga_accs = {}

    def run(self) -> List[Dict[str, Any]]:
        all_rows = []
        
        os.makedirs(os.path.dirname(self.cfg.out_jsonl) if os.path.dirname(self.cfg.out_jsonl) else ".", exist_ok=True)
        jsonl_file = open(self.cfg.out_jsonl, "w", encoding="utf-8")
        
        try:
            for fn in self.cfg.functions:
                if fn not in FUNCTION_NAME_MAPPING:
                    continue

                task_meta = EXPERIMENT_FUNCTION_METADATA.get(fn, {})
                current_lengths = task_meta.get("lengths", self.cfg.lengths)

                for L in current_lengths:
                    train_lines, val_lines, test_lines = self.ds.get(fn, L)
                    target_name = FUNCTION_NAME_MAPPING[fn]
                    
                    is_tabular = target_name in TABULAR_FNS
                    is_boolean = target_name in BOOLEAN_FNS
                    
                    base_models = get_base_models(self.cfg.seed, include_ga=is_boolean, include_tabpfn=True)
                    
                    if is_tabular:
                        parser = TabularDataParser(dataset_name=target_name)
                        X_train, y_train = parser.fit_transform(train_lines)
                        X_val, y_val = parser.transform(val_lines)
                        X_test, y_test = parser.transform(test_lines)
                    else:
                        X_train, y_train = parse_data(train_lines)
                        X_val, y_val = parse_data(val_lines)
                        X_test, y_test = parse_data(test_lines)

                    print(f"{fn} L={L}: train={len(train_lines)}, val={len(val_lines)}, test={len(test_lines)}")

                    param_grids = get_param_grids(tabular=is_tabular, boolean=is_boolean)

                    for model_name in base_models.keys():
                        print(f"  Starting {model_name} (T={self.cfg.num_trials} trials)...", flush=True)
                        try:
                            if model_name not in param_grids:
                                logger.warning(f"{fn} L={L} {model_name}: not found in param_grids, skipping")
                                continue
                            
                            test_accuracies = []
                            val_accuracies = []
                            adaptation_durations = []
                            test_durations = []
                            total_durations = []
                            all_best_params = []
                            
                            for trial in range(self.cfg.num_trials):
                                print(f"    Trial {trial+1}/{self.cfg.num_trials}...", flush=True)
                                trial_seed = self.cfg.seed + trial
                                
                                base_model = get_base_models(trial_seed, include_ga=is_boolean, include_tabpfn=True)[model_name]
                                param_grid = param_grids[model_name]
                                
                                trial_t0 = time.perf_counter()
                                adaptation_t0 = time.perf_counter()
                                
                                if model_name == "tabpfn":
                                    base_model.fit(X_train, y_train)
                                    val_pred = base_model.predict(X_val)
                                    best_val_acc = accuracy_score(y_val, val_pred)
                                    best_params = {}
                                    best_model = base_model
                                    adaptation_duration_ms = int((time.perf_counter() - adaptation_t0) * 1000)
                                    test_t0 = time.perf_counter()
                                    test_pred = best_model.predict(X_test)
                                    test_acc = accuracy_score(y_test, test_pred)
                                    test_duration_ms = int((time.perf_counter() - test_t0) * 1000)
                                elif model_name == "genetic_algorithm":
                                    if is_tabular:
                                        binarizer = GABinarizer()
                                        X_train_ga = binarizer.fit_transform(X_train, parser.numeric_indices, parser.categorical_indices)
                                        X_val_ga = binarizer.transform(X_val, parser.numeric_indices, parser.categorical_indices)
                                    else:
                                        X_train_ga, X_val_ga = X_train, X_val
                                    
                                    best_val_acc = -1
                                    best_params = None
                                    best_model = None
                                    
                                    from itertools import product
                                    keys = list(param_grid.keys())
                                    values = [param_grid[k] for k in keys]
                                    
                                    for combo in product(*values):
                                        params = dict(zip(keys, combo))
                                        params['random_state'] = trial_seed
                                        
                                        model = GeneticAlgorithmClassifier(**params)
                                        model.fit(X_train_ga, y_train, X_val_ga, y_val)
                                        
                                        val_pred = model.predict(X_val_ga)
                                        val_acc = accuracy_score(y_val, val_pred)
                                        
                                        if val_acc > best_val_acc:
                                            best_val_acc = val_acc
                                            best_params = params
                                            best_model = model
                                    
                                    adaptation_duration_ms = int((time.perf_counter() - adaptation_t0) * 1000)
                                    test_t0 = time.perf_counter()
                                    if is_tabular:
                                        X_test_ga = binarizer.transform(X_test, parser.numeric_indices, parser.categorical_indices)
                                    else:
                                        X_test_ga = X_test
                                    test_acc = accuracy_score(y_test, best_model.predict(X_test_ga))
                                    test_duration_ms = int((time.perf_counter() - test_t0) * 1000)
                                    
                                    # Track best GA model across trials (by test accuracy)
                                    task_key = (fn, L)
                                    current_best_acc = self.best_ga_accs.get(task_key, -1.0)
                                    if test_acc >= current_best_acc:
                                        self.best_ga_accs[task_key] = float(test_acc)
                                        # Clone the best node to preserve it
                                        if best_model.best_node_ is not None:
                                            cloned_node = best_model.best_node_.clone()
                                            self.best_ga_models[task_key] = {
                                                "node": cloned_node,
                                                "depth": cloned_node.depth(),
                                                "size": cloned_node.size(),
                                                "expr": cloned_node.to_str(),
                                                "test_acc": float(test_acc),
                                                "params": best_params.copy(),
                                            }
                                            print(f"        New best GA tree: {cloned_node.to_str()} (depth={cloned_node.depth()}, size={cloned_node.size()}, test_acc={test_acc:.4f})", flush=True)
                                else:
                                    grid_search = GridSearchCV(
                                        base_model,
                                        param_grid,
                                        cv=5,
                                        scoring='accuracy',
                                        n_jobs=1,
                                        verbose=0
                                    )
                                    grid_search.fit(X_train, y_train)
                                    best_model = grid_search.best_estimator_
                                    val_acc = accuracy_score(y_val, best_model.predict(X_val))
                                    best_params = grid_search.best_params_
                                    best_val_acc = val_acc
                                    adaptation_duration_ms = int((time.perf_counter() - adaptation_t0) * 1000)

                                    test_t0 = time.perf_counter()
                                    if is_tabular and model_name == "svm":
                                        test_subset_size = min(1000, len(X_test))
                                        test_indices = np.random.RandomState(trial_seed).choice(len(X_test), test_subset_size, replace=False)
                                        test_acc = accuracy_score(y_test[test_indices], best_model.predict(X_test[test_indices]))
                                    else:
                                        test_acc = accuracy_score(y_test, best_model.predict(X_test))
                                    test_duration_ms = int((time.perf_counter() - test_t0) * 1000)

                                    if model_name == "decision_tree":
                                        task_key = (fn, L)
                                        current_best_acc = self.best_decision_tree_accs.get(task_key, -1.0)
                                        if test_acc > current_best_acc:
                                            self.best_decision_tree_accs[task_key] = float(test_acc)
                                            self.best_decision_trees[task_key] = best_model
                                            self.best_decision_tree_infos[task_key] = {
                                                "fn": fn,
                                                "length": L,
                                                "trial": trial,
                                                "val_acc": float(best_val_acc),
                                                "test_acc": float(test_acc),
                                                "best_params": best_params,
                                            }
                                
                                total_duration_ms = int((time.perf_counter() - trial_t0) * 1000)
                                test_accuracies.append(test_acc)
                                val_accuracies.append(best_val_acc)
                                adaptation_durations.append(adaptation_duration_ms)
                                test_durations.append(test_duration_ms)
                                total_durations.append(total_duration_ms)
                                all_best_params.append(best_params)
                                
                                print(
                                    f"      Trial {trial+1}: test_acc={test_acc:.4f}, val_acc={best_val_acc:.4f}, "
                                    f"adapt_ms={adaptation_duration_ms}, test_ms={test_duration_ms}, total_ms={total_duration_ms}",
                                    flush=True
                                )
                            
                            # Compute statistics
                            test_acc_mean = float(np.mean(test_accuracies))
                            test_acc_std = float(np.std(test_accuracies))
                            val_acc_mean = float(np.mean(val_accuracies))
                            val_acc_std = float(np.std(val_accuracies))
                            total_adaptation_duration_ms = int(np.sum(adaptation_durations))
                            total_test_duration_ms = int(np.sum(test_durations))
                            total_wall_clock_duration_ms = int(np.sum(total_durations))
                            
                            # Use first best_params (could use mode if needed)
                            best_params = all_best_params[0]
                            
                            print(f"    {model_name} completed: test_acc={test_acc_mean:.4f}±{test_acc_std:.4f} (mean±std over {self.cfg.num_trials} trials)", flush=True)
                            
                            row = {
                                "fn": fn,
                                "length": L,
                                "model": model_name,
                                "duration_ms": total_wall_clock_duration_ms,
                                "adaptation_duration_ms": total_adaptation_duration_ms,
                                "test_duration_ms": total_test_duration_ms,
                                "total_wall_clock_duration_ms": total_wall_clock_duration_ms,
                                "val_acc": val_acc_mean,
                                "val_acc_std": val_acc_std,
                                "test_acc": test_acc_mean,
                                "test_acc_std": test_acc_std,
                                "best_params": json.dumps(best_params),
                                "best_cv_score": val_acc_mean,
                                "num_trials": self.cfg.num_trials,
                            }
                            
                            # Add tree depth info for GA models (depth of best-performing tree)
                            if model_name == "genetic_algorithm":
                                task_key = (fn, L)
                                best_ga_info = self.best_ga_models.get(task_key)
                                if best_ga_info is not None:
                                    row["tree_depth"] = float(best_ga_info["depth"])
                                    row["tree_size"] = float(best_ga_info["size"])
                                    row["tree_expr"] = best_ga_info["expr"]
                                    print(f"      Best tree: {best_ga_info['expr']}", flush=True)
                                    print(f"      Depth: {best_ga_info['depth']}, Size: {best_ga_info['size']}", flush=True)
                            all_rows.append(row)
                            jsonl_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                            jsonl_file.flush()
                            logger.info(
                                f"{fn} L={L} {model_name}: test_acc={test_acc_mean:.4f}±{test_acc_std:.4f} "
                                f"(mean±std over {self.cfg.num_trials} trials)"
                            )
                        except Exception as e:
                            print(f"    Error processing {model_name}: {str(e)}", flush=True)
                            logger.error(f"{fn} L={L} {model_name}: error - {str(e)}", exc_info=True)
                            continue

            for (fn, length), tree in self.best_decision_trees.items():
                tree_path = os.path.join(current_dir, f"best_decision_tree_{fn}_L{length}.pkl")
                meta_path = os.path.join(current_dir, f"best_decision_tree_{fn}_L{length}_meta.json")
                with open(tree_path, "wb") as f:
                    pickle.dump(tree, f)
                meta = dict(self.best_decision_tree_infos[(fn, length)])
                meta["model_type"] = "decision_tree"
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(meta, f, ensure_ascii=False)
                logger.info(f"Best decision tree for {fn} L={length} saved to {tree_path} with metadata {meta_path}")
        finally:
            jsonl_file.close()

        return all_rows


class BenchmarkRunner:
    """Validation-selected baseline benchmark runner.

    This is the runner used by `main()`. The older `BaselineRunner` above is left
    in place for backward reference, but this runner has the benchmark behavior we
    want for the tabular comparison table.
    """

    SCALED_MODELS = {"svm", "logistic_regression", "mlp"}

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.ds = DatasetStore(cfg)
        self.best_decision_trees: Dict[Tuple[str, int], Any] = {}
        self.best_decision_tree_infos: Dict[Tuple[str, int], Dict[str, Any]] = {}

    @staticmethod
    def _subset_arrays(
        X: np.ndarray,
        y: np.ndarray,
        max_rows: Optional[int],
        seed: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if max_rows is None or len(y) <= max_rows:
            return X, y
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(y), max_rows, replace=False)
        return X[idx], y[idx]

    @staticmethod
    def _scaled_arrays(
        X_train: np.ndarray,
        X_val: np.ndarray,
        X_test: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if X_train.size == 0:
            return X_train, X_val, X_test
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0)
        std[std < 1e-8] = 1.0
        return (X_train - mean) / std, (X_val - mean) / std, (X_test - mean) / std

    @staticmethod
    def _selection_split(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, str]:
        if len(y_val) > 0:
            return X_val, y_val, "val"
        return X_train, y_train, "train"

    def _write_row(self, jsonl_file, rows: List[Dict[str, Any]], row: Dict[str, Any]) -> None:
        rows.append(row)
        jsonl_file.write(json.dumps(row, ensure_ascii=False) + "\n")
        jsonl_file.flush()

    def _error_row(
        self,
        fn: str,
        target_name: str,
        length: int,
        model_name: str,
        error: str,
    ) -> Dict[str, Any]:
        return {
            "fn": fn,
            "target_name": target_name,
            "length": length,
            "model": model_name,
            "status": "error",
            "error": error,
            "duration_ms": 0,
            "adaptation_duration_ms": 0,
            "test_duration_ms": 0,
            "total_wall_clock_duration_ms": 0,
            "val_acc": None,
            "val_acc_std": None,
            "test_acc": None,
            "test_acc_std": None,
            "best_params": "{}",
            "best_cv_score": None,
            "num_trials": self.cfg.num_trials,
            "tabular_representation": self.cfg.tabular_representation,
            "train_size": self.cfg.train_size,
            "val_size": self.cfg.val_size,
            "test_size": self.cfg.test_size,
            "seed": self.cfg.seed,
            "selection_split": "",
            "mean_fit_count": 0,
        }

    def _fit_select_model(
        self,
        model_name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_select: np.ndarray,
        y_select: np.ndarray,
        trial_seed: int,
        param_grid: Dict[str, List[Any]],
    ) -> Tuple[Any, Dict[str, Any], float, int]:
        best_model = None
        best_params: Dict[str, Any] = {}
        best_score = -1.0
        fit_count = 0

        for params in iter_param_grid(param_grid):
            model = get_base_models(
                trial_seed,
                include_ga=self.cfg.include_ga,
                include_tabpfn=self.cfg.include_tabpfn,
            ).get(model_name)
            if model is None:
                raise RuntimeError(f"Model '{model_name}' is unavailable in this environment.")
            if params:
                model.set_params(**params)
            model.fit(X_train, y_train)
            score = accuracy_score(y_select, model.predict(X_select))
            fit_count += 1
            if score > best_score:
                best_score = float(score)
                best_params = dict(params)
                best_model = model

        if best_model is None:
            raise RuntimeError(f"No successful fit for model '{model_name}'.")
        return best_model, best_params, best_score, fit_count

    def _fit_select_ga(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_select: np.ndarray,
        y_select: np.ndarray,
        trial_seed: int,
        param_grid: Dict[str, List[Any]],
    ) -> Tuple[Any, Dict[str, Any], float, int]:
        best_model = None
        best_params: Dict[str, Any] = {}
        best_score = -1.0
        fit_count = 0

        for params in iter_param_grid(param_grid):
            params = dict(params)
            params["random_state"] = trial_seed
            model = GeneticAlgorithmClassifier(**params)
            model.fit(X_train, y_train, X_select, y_select)
            score = accuracy_score(y_select, model.predict(X_select))
            fit_count += 1
            if score > best_score:
                best_score = float(score)
                best_params = dict(params)
                best_model = model

        if best_model is None:
            raise RuntimeError("No successful fit for genetic_algorithm.")
        return best_model, best_params, best_score, fit_count

    def _requested_models(self) -> List[str]:
        models = list(self.cfg.models)
        if self.cfg.include_ga and "genetic_algorithm" not in models:
            models.append("genetic_algorithm")
        if self.cfg.include_tabpfn and "tabpfn" not in models:
            models.append("tabpfn")
        return models

    def _prepare_data(
        self,
        fn: str,
        length: int,
        target_name: str,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        train_lines, val_lines, test_lines = self.ds.get(fn, length)
        if target_name in TABULAR_FNS:
            parser = TabularDataParser(dataset_name=target_name)
            X_train, y_train = parser.fit_transform(train_lines)
            X_val, y_val = parser.transform(val_lines)
            X_test, y_test = parser.transform(test_lines)
        else:
            X_train, y_train = parse_data(train_lines)
            X_val, y_val = parse_data(val_lines)
            X_test, y_test = parse_data(test_lines)

        X_train, y_train = self._subset_arrays(X_train, y_train, self.cfg.max_train_rows, self.cfg.seed)
        X_test, y_test = self._subset_arrays(X_test, y_test, self.cfg.max_test_rows, self.cfg.seed + 100_000)
        return X_train, y_train, X_val, y_val, X_test, y_test

    def run(self) -> List[Dict[str, Any]]:
        all_rows: List[Dict[str, Any]] = []
        os.makedirs(os.path.dirname(self.cfg.out_jsonl) if os.path.dirname(self.cfg.out_jsonl) else ".", exist_ok=True)

        with open(self.cfg.out_jsonl, "w", encoding="utf-8") as jsonl_file:
            for fn in self.cfg.functions:
                if fn not in FUNCTION_NAME_MAPPING:
                    logger.warning("Unknown function id %s; skipping", fn)
                    continue

                target_name = FUNCTION_NAME_MAPPING[fn]
                task_meta = EXPERIMENT_FUNCTION_METADATA.get(fn, {})
                current_lengths = task_meta.get("lengths", self.cfg.lengths)
                is_boolean = target_name in BOOLEAN_FNS
                param_grids = get_param_grids(boolean=is_boolean)

                for length in current_lengths:
                    try:
                        X_train, y_train, X_val, y_val, X_test, y_test = self._prepare_data(fn, length, target_name)
                    except Exception as exc:
                        for model_name in self._requested_models():
                            self._write_row(
                                jsonl_file,
                                all_rows,
                                self._error_row(fn, target_name, length, model_name, f"data error: {exc}"),
                            )
                        continue

                    X_train_scaled, X_val_scaled, X_test_scaled = self._scaled_arrays(X_train, X_val, X_test)
                    print(
                        f"{fn} ({target_name}) L={length}: train={len(y_train)}, "
                        f"val={len(y_val)}, test={len(y_test)}",
                        flush=True,
                    )

                    for model_name in self._requested_models():
                        if model_name not in param_grids:
                            self._write_row(
                                jsonl_file,
                                all_rows,
                                self._error_row(fn, target_name, length, model_name, "unknown model"),
                            )
                            continue
                        if model_name == "xgboost" and XGBClassifier is None:
                            self._write_row(
                                jsonl_file,
                                all_rows,
                                self._error_row(fn, target_name, length, model_name, "xgboost is not installed"),
                            )
                            continue
                        if model_name == "tabpfn" and TabPFNClassifier is None:
                            self._write_row(
                                jsonl_file,
                                all_rows,
                                self._error_row(fn, target_name, length, model_name, "tabpfn is not installed"),
                            )
                            continue

                        print(f"  Starting {model_name} ({self.cfg.num_trials} trials)...", flush=True)
                        test_accuracies: List[float] = []
                        val_accuracies: List[float] = []
                        adaptation_durations: List[int] = []
                        test_durations: List[int] = []
                        total_durations: List[int] = []
                        fit_counts: List[int] = []
                        best_params_by_trial: List[Dict[str, Any]] = []
                        selection_split = "val" if len(y_val) > 0 else "train"

                        try:
                            for trial in range(self.cfg.num_trials):
                                trial_seed = self.cfg.seed + trial
                                use_scaled = model_name in self.SCALED_MODELS
                                Xtr = X_train_scaled if use_scaled else X_train
                                Xv = X_val_scaled if use_scaled else X_val
                                Xte = X_test_scaled if use_scaled else X_test
                                X_select, y_select, selection_split = self._selection_split(Xtr, y_train, Xv, y_val)

                                trial_t0 = time.perf_counter()
                                adaptation_t0 = time.perf_counter()

                                if model_name == "genetic_algorithm":
                                    binarizer = GABinarizer()
                                    X_train_ga = binarizer.fit_transform(X_train)
                                    X_val_ga = binarizer.transform(X_val)
                                    X_select_ga, y_select_ga, selection_split = self._selection_split(
                                        X_train_ga, y_train, X_val_ga, y_val
                                    )
                                    best_model, best_params, best_val_acc, fit_count = self._fit_select_ga(
                                        X_train_ga,
                                        y_train,
                                        X_select_ga,
                                        y_select_ga,
                                        trial_seed,
                                        param_grids[model_name],
                                    )
                                    Xte = binarizer.transform(X_test)
                                else:
                                    best_model, best_params, best_val_acc, fit_count = self._fit_select_model(
                                        model_name,
                                        Xtr,
                                        y_train,
                                        X_select,
                                        y_select,
                                        trial_seed,
                                        param_grids[model_name],
                                    )

                                adaptation_ms = int((time.perf_counter() - adaptation_t0) * 1000)
                                test_t0 = time.perf_counter()
                                test_acc = float(accuracy_score(y_test, best_model.predict(Xte)))
                                test_ms = int((time.perf_counter() - test_t0) * 1000)
                                total_ms = int((time.perf_counter() - trial_t0) * 1000)

                                if model_name == "decision_tree":
                                    key = (fn, length)
                                    previous = self.best_decision_tree_infos.get(key, {}).get("test_acc", -1.0)
                                    if test_acc > previous:
                                        self.best_decision_trees[key] = best_model
                                        self.best_decision_tree_infos[key] = {
                                            "fn": fn,
                                            "target_name": target_name,
                                            "length": length,
                                            "trial": trial,
                                            "val_acc": float(best_val_acc),
                                            "test_acc": test_acc,
                                            "best_params": best_params,
                                        }

                                test_accuracies.append(test_acc)
                                val_accuracies.append(float(best_val_acc))
                                adaptation_durations.append(adaptation_ms)
                                test_durations.append(test_ms)
                                total_durations.append(total_ms)
                                fit_counts.append(fit_count)
                                best_params_by_trial.append(best_params)
                                print(
                                    f"    Trial {trial + 1}: test={test_acc:.4f}, "
                                    f"{selection_split}={best_val_acc:.4f}, fits={fit_count}, "
                                    f"adapt_ms={adaptation_ms}, test_ms={test_ms}",
                                    flush=True,
                                )

                            row = {
                                "fn": fn,
                                "target_name": target_name,
                                "length": length,
                                "model": model_name,
                                "status": "ok",
                                "error": "",
                                "duration_ms": int(np.sum(total_durations)),
                                "adaptation_duration_ms": int(np.sum(adaptation_durations)),
                                "test_duration_ms": int(np.sum(test_durations)),
                                "total_wall_clock_duration_ms": int(np.sum(total_durations)),
                                "val_acc": float(np.mean(val_accuracies)),
                                "val_acc_std": float(np.std(val_accuracies)),
                                "test_acc": float(np.mean(test_accuracies)),
                                "test_acc_std": float(np.std(test_accuracies)),
                                "best_params": json.dumps(best_params_by_trial[0] if best_params_by_trial else {}),
                                "best_cv_score": float(np.mean(val_accuracies)),
                                "num_trials": self.cfg.num_trials,
                                "tabular_representation": self.cfg.tabular_representation,
                                "train_size": len(y_train),
                                "val_size": len(y_val),
                                "test_size": len(y_test),
                                "seed": self.cfg.seed,
                                "selection_split": selection_split,
                                "mean_fit_count": float(np.mean(fit_counts)),
                            }
                            self._write_row(jsonl_file, all_rows, row)
                            logger.info(
                                "%s L=%s %s: test_acc=%.4f+/-%.4f",
                                fn,
                                length,
                                model_name,
                                row["test_acc"],
                                row["test_acc_std"],
                            )
                        except Exception as exc:
                            logger.error("%s L=%s %s failed", fn, length, model_name, exc_info=True)
                            self._write_row(
                                jsonl_file,
                                all_rows,
                                self._error_row(fn, target_name, length, model_name, str(exc)),
                            )

        if self.cfg.save_model_artifacts:
            for (fn, length), tree in self.best_decision_trees.items():
                tree_path = os.path.join(current_dir, f"best_decision_tree_{fn}_L{length}.pkl")
                meta_path = os.path.join(current_dir, f"best_decision_tree_{fn}_L{length}_meta.json")
                with open(tree_path, "wb") as f:
                    pickle.dump(tree, f)
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(self.best_decision_tree_infos[(fn, length)], f, ensure_ascii=False)

        return all_rows


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        with open(path, "w", newline="", encoding="utf-8") as f:
            f.write("")
        return
    preferred = [
        "fn", "target_name", "length", "model", "status", "error",
        "tabular_representation", "train_size", "val_size", "test_size", "seed",
        "duration_ms", "adaptation_duration_ms", "test_duration_ms", "total_wall_clock_duration_ms",
        "val_acc", "val_acc_std", "test_acc", "test_acc_std",
        "best_params", "best_cv_score", "num_trials", "selection_split", "mean_fit_count",
        "tree_depth", "tree_size", "tree_expr",
    ]
    extras = sorted({key for row in rows for key in row.keys()} - set(preferred))
    fieldnames = [key for key in preferred if any(key in row for row in rows)] + extras
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Baseline ML models runner")
    p.add_argument("--functions", nargs="*", help="Function IDs (e.g., fn_a fn_b ...)")
    p.add_argument("--lengths", nargs="*", type=int, help="Sequence lengths")
    p.add_argument("--models", nargs="*", help="Model names to run. Defaults to the core tabular baseline set.")
    p.add_argument("--train-size", type=int, help="Train size (default: 100)")
    p.add_argument("--val-size", type=int, help="Validation size (default: 100)")
    p.add_argument("--test-size", type=int, help="Test size (default: 10000)")
    p.add_argument("--seed", type=int, help="Global seed (default: 42)")
    p.add_argument("--num-trials", type=int, help="Number of trials for averaging (default: 10)")
    p.add_argument("--dataset-dir", help="Dataset split/cache directory")
    p.add_argument(
        "--tabular-representation",
        choices=["obfuscated", "semantic"],
        help="Use obfuscated or semantic/named tabular dataset rows.",
    )
    p.add_argument("--out-jsonl", help="Output JSONL path")
    p.add_argument("--out-csv", help="Output CSV path")
    p.add_argument("--include-ga", action="store_true", help="Include the symbolic genetic-programming baseline.")
    p.add_argument("--include-tabpfn", action="store_true", help="Include TabPFN if installed.")
    p.add_argument("--max-train-rows", type=int, help="Optional cap for fitting rows per split.")
    p.add_argument("--max-test-rows", type=int, help="Optional cap for test evaluation rows.")
    p.add_argument("--save-model-artifacts", action="store_true", help="Persist selected tree artifacts.")

    args = p.parse_args()
    cfg = Config()

    if args.functions: cfg.functions = args.functions
    if args.lengths: cfg.lengths = args.lengths
    if args.models: cfg.models = args.models
    if args.train_size: cfg.train_size = args.train_size
    if args.val_size is not None: cfg.val_size = args.val_size
    if args.test_size: cfg.test_size = args.test_size
    if args.seed is not None: cfg.seed = args.seed
    if args.num_trials is not None: cfg.num_trials = args.num_trials
    if args.dataset_dir: cfg.dataset_dir = args.dataset_dir
    if args.tabular_representation: cfg.tabular_representation = args.tabular_representation
    if cfg.tabular_representation != "obfuscated":
        cfg.dataset_dir = os.path.join(cfg.dataset_dir, f"tabular_representation_{cfg.tabular_representation}")
    if args.out_jsonl: cfg.out_jsonl = args.out_jsonl
    if args.out_csv: cfg.out_csv = args.out_csv
    if args.include_ga: cfg.include_ga = True
    if args.include_tabpfn: cfg.include_tabpfn = True
    if args.max_train_rows is not None: cfg.max_train_rows = args.max_train_rows
    if args.max_test_rows is not None: cfg.max_test_rows = args.max_test_rows
    if args.save_model_artifacts: cfg.save_model_artifacts = True

    return cfg


def main():
    cfg = parse_args()
    runner = BenchmarkRunner(cfg)
    rows = runner.run()
    write_csv(cfg.out_csv, rows)
    logger.info(f"Results written to {cfg.out_jsonl} and {cfg.out_csv}")


if __name__ == "__main__":
    main()

