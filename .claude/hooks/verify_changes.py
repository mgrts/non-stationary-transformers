#!/usr/bin/env python3
"""Stop hook: dependency-free static verification of changed Python sources.

This repo has no pytest suite and its runtime deps (torch, sklearn, pandas,
mlflow, dotenv) are not installed in every environment, so the verification gate
is intentionally import-free. When the turn touched .py files under src/, it:

  1. byte-compiles every changed file (``py_compile``) - catches syntax errors;
  2. AST-checks that every ``from src.config import (...)`` name is actually
     defined in src/config.py - catches the import drift that has historically
     bitten this repo (renamed/removed config constants).

On failure it blocks the stop once (exit 2) so the agent sees and fixes the
problem; ``stop_hook_active`` prevents an infinite loop. Disable by removing the
"Stop" entry from .claude/settings.json.
"""

import ast
import json
import os
import py_compile
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _gitutil import status_paths  # noqa: E402


def config_defined_names(config_path):
    """Top-level names assigned or def'd in src/config.py."""
    names = set()
    try:
        tree = ast.parse(open(config_path).read(), config_path)
    except (OSError, SyntaxError):
        return names
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    names.add(t.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
    return names


def missing_config_imports(py_file, defined):
    """Names imported from src.config in py_file that are not defined there."""
    try:
        tree = ast.parse(open(py_file).read(), py_file)
    except (OSError, SyntaxError):
        return []
    missing = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "src.config":
            for alias in node.names:
                if alias.name != "*" and alias.name not in defined:
                    missing.append(alias.name)
    return missing


def main() -> int:
    try:
        data = json.load(sys.stdin)
    except Exception:
        data = {}

    # Avoid loops: if the previous stop was already triggered by this hook, let it stop.
    if data.get("stop_hook_active"):
        return 0

    project_dir = os.environ.get("CLAUDE_PROJECT_DIR") or os.getcwd()
    touched = [p for p in status_paths(project_dir) if p.endswith(".py") and p.startswith("src/")]
    if not touched:
        return 0

    errors = []

    # 1. Byte-compile each changed source file.
    for rel in touched:
        abs_path = os.path.join(project_dir, rel)
        if not os.path.isfile(abs_path):
            continue
        try:
            py_compile.compile(abs_path, doraise=True)
        except py_compile.PyCompileError as exc:
            errors.append(f"{rel}: compile error: {exc.msg.strip()}")

    # 2. Cross-check src.config imports against what config.py defines.
    config_path = os.path.join(project_dir, "src", "config.py")
    if os.path.isfile(config_path):
        defined = config_defined_names(config_path)
        for rel in touched:
            abs_path = os.path.join(project_dir, rel)
            if not os.path.isfile(abs_path):
                continue
            for name in missing_config_imports(abs_path, defined):
                errors.append(f"{rel}: imports undefined config name '{name}'")

    if not errors:
        print(f"[verify-changes] {len(touched)} changed src file(s) compile; config imports OK.")
        return 0

    sys.stderr.write("⛔ verify-changes found problems in your edits (auto-run on stop):\n")
    for e in errors[:25]:
        sys.stderr.write(f"  - {e}\n")
    if len(errors) > 25:
        sys.stderr.write(f"  ... and {len(errors) - 25} more\n")
    sys.stderr.write(
        '\nFix these before finishing, or remove the "Stop" hook in '
        ".claude/settings.json to disable this gate.\n"
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
