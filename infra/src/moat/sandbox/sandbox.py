"""
Sandbox for safe execution of untrusted strategy code.

Provides a restricted execution environment that blocks dangerous
operations like file system access, network calls, and imports.
"""

import ast
import logging
from typing import Any, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class SandboxError(Exception):
    """Raised when sandbox detects unsafe code."""

    pass


class CodeValidator(ast.NodeVisitor):
    """AST visitor that validates code safety."""

    # Blocked built-in names
    BLOCKED_BUILTINS = {
        "eval",
        "exec",
        "compile",
        "open",
        "input",
        "__import__",
        "globals",
        "locals",
        "vars",
        "dir",
        "getattr",
        "setattr",
        "delattr",
        "hasattr",
        "type",
        "isinstance",
        "issubclass",
        "callable",
        "breakpoint",
        "memoryview",
        "property",
        "staticmethod",
        "classmethod",
        "super",
    }

    # Blocked module names
    BLOCKED_MODULES = {
        "os",
        "sys",
        "subprocess",
        "shutil",
        "pathlib",
        "io",
        "socket",
        "requests",
        "urllib",
        "http",
        "ftplib",
        "smtplib",
        "pickle",
        "shelve",
        "marshal",
        "importlib",
        "builtins",
        "__builtins__",
        "ctypes",
        "multiprocessing",
        "threading",
        "asyncio",
        "concurrent",
    }

    # Blocked attribute access patterns
    BLOCKED_ATTRIBUTES = {
        "__class__",
        "__bases__",
        "__mro__",
        "__subclasses__",
        "__code__",
        "__globals__",
        "__dict__",
        "__module__",
        "__import__",
        "__builtins__",
        "__file__",
        "__name__",
        "__loader__",
        "__spec__",
    }

    def __init__(self) -> None:
        self.errors: list[str] = []

    def visit_Import(self, node: ast.Import) -> None:
        """Block all imports."""
        for alias in node.names:
            module_name = alias.name.split(".")[0]
            self.errors.append(f"Import not allowed: {alias.name}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Block all from...import statements."""
        module = node.module or ""
        self.errors.append(f"Import not allowed: from {module}")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        """Check function calls for blocked builtins."""
        if isinstance(node.func, ast.Name):
            if node.func.id in self.BLOCKED_BUILTINS:
                self.errors.append(f"Blocked builtin: {node.func.id}")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        """Block access to dangerous attributes."""
        if node.attr in self.BLOCKED_ATTRIBUTES:
            self.errors.append(f"Blocked attribute: {node.attr}")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        """Block access to dangerous names."""
        if node.id in self.BLOCKED_MODULES:
            self.errors.append(f"Blocked module reference: {node.id}")
        self.generic_visit(node)

    def validate(self, code: str) -> list[str]:
        """Validate code and return list of errors."""
        self.errors = []
        try:
            tree = ast.parse(code)
            self.visit(tree)
        except SyntaxError as e:
            self.errors.append(f"Syntax error: {e}")
        return self.errors


class Sandbox:
    """Restricted execution environment for strategy code.

    Executes user-provided code in a sandboxed environment with:
    - No imports allowed
    - No file system access
    - No network access
    - Only whitelisted functions available
    """

    def __init__(self, allowed_globals: Optional[dict[str, Any]] = None) -> None:
        """Initialize sandbox.

        Args:
            allowed_globals: Dict of names to make available in sandbox.
                           If None, uses safe defaults.
        """
        self._validator = CodeValidator()
        self._allowed_globals = allowed_globals or {}

    def validate(self, code: str) -> tuple[bool, list[str]]:
        """Validate code without executing.

        Args:
            code: Python code string to validate.

        Returns:
            Tuple of (is_valid, list of error messages).
        """
        errors = self._validator.validate(code)
        return len(errors) == 0, errors

    def execute(
        self,
        code: str,
        local_vars: Optional[dict[str, Any]] = None,
    ) -> Any:
        """Execute code in sandboxed environment.

        Args:
            code: Python code to execute.
            local_vars: Variables to make available to the code.

        Returns:
            Result of the last expression, or None.

        Raises:
            SandboxError: If code is unsafe or execution fails.
        """
        # Validate first
        is_valid, errors = self.validate(code)
        if not is_valid:
            raise SandboxError(f"Unsafe code detected: {'; '.join(errors)}")

        # Prepare execution environment
        safe_builtins = self._get_safe_builtins()
        exec_globals = {
            "__builtins__": safe_builtins,
            **self._allowed_globals,
        }
        exec_locals = local_vars or {}

        # Execute in restricted environment
        try:
            # Compile and execute
            compiled = compile(code, "<sandbox>", "exec")
            exec(compiled, exec_globals, exec_locals)

            # Return 'result' if defined, else None
            return exec_locals.get("result", None)

        except Exception as e:
            raise SandboxError(f"Execution error: {e}") from e

    def _get_safe_builtins(self) -> dict[str, Any]:
        """Get restricted set of safe builtins."""
        import builtins

        safe = {
            # Math
            "abs": builtins.abs,
            "round": builtins.round,
            "min": builtins.min,
            "max": builtins.max,
            "sum": builtins.sum,
            "pow": builtins.pow,
            "divmod": builtins.divmod,
            # Type conversion
            "int": builtins.int,
            "float": builtins.float,
            "bool": builtins.bool,
            "str": builtins.str,
            # Iteration
            "len": builtins.len,
            "range": builtins.range,
            "enumerate": builtins.enumerate,
            "zip": builtins.zip,
            "map": builtins.map,
            "filter": builtins.filter,
            "sorted": builtins.sorted,
            "reversed": builtins.reversed,
            # Aggregation
            "all": builtins.all,
            "any": builtins.any,
            # Comparison
            "True": True,
            "False": False,
            "None": None,
            # List/tuple/dict (constructors only)
            "list": builtins.list,
            "tuple": builtins.tuple,
            "dict": builtins.dict,
            "set": builtins.set,
        }
        return safe


class StrategyRunner:
    """High-level interface for running sandboxed strategies.

    Combines the Sandbox with allowed financial primitives to
    execute user strategy code safely.
    """

    def __init__(self, sandbox: Optional[Sandbox] = None) -> None:
        """Initialize strategy runner.

        Args:
            sandbox: Sandbox instance to use. Creates default if None.
        """
        from moat.sandbox.dsl_wrapper import get_dsl_globals

        dsl_globals = get_dsl_globals()
        self._sandbox = sandbox or Sandbox(allowed_globals=dsl_globals)

    def run(
        self,
        code: str,
        data: pd.DataFrame,
    ) -> pd.Series:
        """Execute strategy code and return signals.

        The code should define 'result' as the signal series.
        Data columns are available as: open, high, low, close, volume.

        Args:
            code: Strategy code string.
            data: OHLCV DataFrame.

        Returns:
            Signal series (-1 to 1).

        Raises:
            SandboxError: If code is unsafe or fails.
        """
        # Prepare data variables
        local_vars = {
            "open": data["open"],
            "high": data["high"],
            "low": data["low"],
            "close": data["close"],
            "volume": data["volume"],
            "data": data,
        }

        # Execute in sandbox
        result = self._sandbox.execute(code, local_vars)

        # Validate result
        if result is None:
            raise SandboxError("Strategy must define 'result' variable")

        if not isinstance(result, pd.Series):
            # Try to convert
            result = pd.Series(result, index=data.index)

        # Clip to valid signal range
        result = result.clip(-1, 1)

        return result

    def validate(self, code: str) -> tuple[bool, list[str]]:
        """Validate strategy code without executing.

        Args:
            code: Strategy code to validate.

        Returns:
            Tuple of (is_valid, errors).
        """
        return self._sandbox.validate(code)
