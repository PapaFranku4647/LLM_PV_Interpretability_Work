from __future__ import annotations

import sys
from pathlib import Path
import unittest


PROGRAM_SYNTHESIS_DIR = Path(__file__).resolve().parents[1]
if str(PROGRAM_SYNTHESIS_DIR) not in sys.path:
    sys.path.insert(0, str(PROGRAM_SYNTHESIS_DIR))

from code_normalizer import (  # noqa: E402
    normalize_generated_code,
    sanitize_generated_code,
    strip_comment_tokens,
    strip_docstrings_ast,
)


class CodeNormalizerTests(unittest.TestCase):
    def test_strips_full_line_and_inline_comments(self) -> None:
        code = """
        # top-level comment
        def f(x):
            # inside function comment
            return x + 1  # trailing comment
        """
        stripped = strip_comment_tokens(code)
        self.assertNotIn("top-level comment", stripped)
        self.assertNotIn("inside function comment", stripped)
        self.assertNotIn("trailing comment", stripped)

    def test_preserves_hash_inside_string_literals(self) -> None:
        code = """
        def f(x):
            token = "#not-a-comment"
            return 1 if token.startswith("#") else 0
        """
        stripped = strip_comment_tokens(code)
        self.assertIn("#not-a-comment", stripped)

    def test_removes_module_docstring(self) -> None:
        code = '''
        """MODULE_DOC_UNIQUE"""
        def f(x):
            return x
        '''
        stripped = strip_docstrings_ast(code)
        self.assertNotIn("MODULE_DOC_UNIQUE", stripped)
        self.assertIn("def f", stripped)

    def test_removes_function_docstring(self) -> None:
        code = '''
        def f(x):
            """FUNC_DOC_UNIQUE"""
            return x
        '''
        stripped = strip_docstrings_ast(code)
        self.assertNotIn("FUNC_DOC_UNIQUE", stripped)
        self.assertIn("return x", stripped)

    def test_removes_class_docstring(self) -> None:
        code = '''
        class Demo:
            """CLASS_DOC_UNIQUE"""
            def f(self, x):
                return x
        '''
        stripped = strip_docstrings_ast(code)
        self.assertNotIn("CLASS_DOC_UNIQUE", stripped)
        self.assertIn("class Demo", stripped)

    def test_preserves_non_docstring_string_literals(self) -> None:
        code = """
        def f(x):
            label = "KEEP_STRING_LITERAL"
            return label
        """
        stripped = strip_docstrings_ast(code)
        self.assertIn("KEEP_STRING_LITERAL", stripped)

    def test_unwraps_fenced_code_blocks(self) -> None:
        code = """
        ```python
        def f(x):
            return x
        ```
        """
        normalized = normalize_generated_code(code)
        self.assertEqual(normalized, "def f(x):\n    return x")

    def test_sanitize_is_idempotent(self) -> None:
        code = '''
        ```python
        """MODULE_DOC"""
        # top
        def f(x):
            """FUNC_DOC"""
            # local
            return 1 if x else 0  # trailing
        ```
        '''
        once = sanitize_generated_code(code)
        twice = sanitize_generated_code(once)
        self.assertEqual(once, twice)


if __name__ == "__main__":
    unittest.main()
