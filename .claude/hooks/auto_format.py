#!/usr/bin/env python3
"""PostToolUse hook (Edit|Write|MultiEdit): run isort + black on an edited .py file.

Keeps the working tree consistent with the repo's pre-commit config (isort & black,
line-length 99) so diffs stay clean and `git commit` never trips the formatting hooks.
Best-effort: if black/isort are not installed it silently skips, and it always exits 0
so it never blocks an edit.
"""

import json
import os
import subprocess
import sys

LINE_LENGTH = "99"


def main() -> int:
    try:
        data = json.load(sys.stdin)
    except Exception:
        return 0

    file_path = (data.get("tool_input") or {}).get("file_path", "")
    if not file_path or not file_path.endswith(".py"):
        return 0

    project_dir = os.environ.get("CLAUDE_PROJECT_DIR") or os.getcwd()
    abs_file = os.path.abspath(file_path)
    abs_proj = os.path.abspath(project_dir)
    try:
        inside = os.path.commonpath([abs_file, abs_proj]) == abs_proj
    except ValueError:
        inside = False
    if not inside or not os.path.isfile(abs_file):
        return 0

    # Prefer a project venv if present, else fall back to PATH.
    venv_bin = os.path.join(project_dir, ".venv", "bin")

    def tool(name):
        p = os.path.join(venv_bin, name)
        return p if os.path.exists(p) else name

    applied = []
    for name, args in (
        ("isort", ["--profile", "black", "--line-length", LINE_LENGTH]),
        ("black", ["--line-length", LINE_LENGTH]),
    ):
        try:
            subprocess.run(
                [tool(name), *args, abs_file],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            applied.append(name)
        except FileNotFoundError:
            pass  # tool not installed - skip silently

    if applied:
        print(
            f"[auto-format] {' + '.join(applied)} applied to "
            f"{os.path.relpath(abs_file, abs_proj)}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
