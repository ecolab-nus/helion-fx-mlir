from __future__ import annotations

from pathlib import Path
import re

REPO_ROOT = Path(__file__).resolve().parents[1]

DENYLIST = [
    r"ecolab-nus",
    r"loom-dataflow",
    r"pytorch-labs/helion",
    r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}",
]

ALLOWED_PATH_SUFFIXES = {
    "LICENSE",
}


def _is_allowed(path: Path) -> bool:
    return any(str(path).endswith(suffix) for suffix in ALLOWED_PATH_SUFFIXES)


def test_no_identifying_strings_in_repo_text() -> None:
    violations: list[str] = []
    patterns = [re.compile(p, re.IGNORECASE) for p in DENYLIST]

    for path in REPO_ROOT.rglob("*"):
        if not path.is_file() or _is_allowed(path):
            continue
        if ".git" in path.parts or "__pycache__" in path.parts or ".pytest_cache" in path.parts:
            continue
        if path.name == "test_anonymization_guard.py":
            continue
        if path.suffix.lower() not in {".py", ".md", ".toml", ".txt", ".mlir"}:
            continue

        text = path.read_text(encoding="utf-8", errors="ignore")
        for pattern in patterns:
            if pattern.search(text):
                violations.append(f"{path.relative_to(REPO_ROOT)} matches {pattern.pattern}")

    assert not violations, "Found anonymization violations:\n" + "\n".join(sorted(violations))
