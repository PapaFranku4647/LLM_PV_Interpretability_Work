from __future__ import annotations

import ast
import io
import re
import textwrap
import tokenize
from typing import List


_OUTER_FENCE_RE = re.compile(
    r"^\s*```(?:python|py)?\s*\n(?P<code>[\s\S]*?)\n\s*```\s*$",
    flags=re.IGNORECASE,
)


def normalize_generated_code(code_str: str) -> str:
    normalized = textwrap.dedent(code_str or "").strip()
    if not normalized:
        return ""

    m = _OUTER_FENCE_RE.match(normalized)
    if m:
        normalized = m.group("code")
    elif normalized.startswith("```") and normalized.endswith("```"):
        normalized = re.sub(
            r"^\s*```(?:python|py)?\s*|\s*```\s*$",
            "",
            normalized,
            flags=re.IGNORECASE | re.DOTALL,
        )
    return normalized.strip()


def strip_comment_tokens(code_str: str) -> str:
    if not code_str:
        return ""
    try:
        out_tokens = []
        for tok in tokenize.generate_tokens(io.StringIO(code_str).readline):
            if tok.type == tokenize.COMMENT:
                continue
            out_tokens.append(tok)
        return tokenize.untokenize(out_tokens)
    except Exception:
        return code_str


def _is_docstring_expr(stmt: ast.stmt) -> bool:
    if not isinstance(stmt, ast.Expr):
        return False
    value = stmt.value
    if isinstance(value, ast.Constant):
        return isinstance(value.value, str)
    return isinstance(value, ast.Str)


class _DocstringStripper(ast.NodeTransformer):
    @staticmethod
    def _strip_docstring(body: List[ast.stmt]) -> List[ast.stmt]:
        if body and _is_docstring_expr(body[0]):
            return body[1:]
        return body

    def visit_Module(self, node: ast.Module) -> ast.Module:
        node.body = self._strip_docstring(node.body)
        self.generic_visit(node)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        node.body = self._strip_docstring(node.body)
        self.generic_visit(node)
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AsyncFunctionDef:
        node.body = self._strip_docstring(node.body)
        self.generic_visit(node)
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        node.body = self._strip_docstring(node.body)
        self.generic_visit(node)
        return node


def strip_docstrings_ast(code_str: str) -> str:
    normalized = textwrap.dedent(code_str or "").strip()
    if not normalized:
        return ""
    try:
        tree = ast.parse(normalized)
    except Exception:
        return normalized
    tree = _DocstringStripper().visit(tree)
    ast.fix_missing_locations(tree)
    try:
        return ast.unparse(tree)
    except Exception:
        return normalized


def sanitize_generated_code(code_str: str) -> str:
    normalized = normalize_generated_code(code_str)
    without_comments = strip_comment_tokens(normalized)
    return strip_docstrings_ast(without_comments)
