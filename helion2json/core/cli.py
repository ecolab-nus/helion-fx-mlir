from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Sequence

import torch

from .capture import export_kernel


def _parse_entry(entry: str) -> tuple[str, str]:
    if ":" not in entry:
        raise argparse.ArgumentTypeError("Entry must be formatted as 'module:function'")
    module_name, func_name = entry.split(":", 1)
    if not module_name or not func_name:
        raise argparse.ArgumentTypeError("Invalid entry specification")
    return module_name, func_name


def _eval_arg(expr: str, scope: dict[str, object]) -> object:
    try:
        return eval(expr, scope, scope)
    except Exception as exc:  # noqa: BLE001
        raise argparse.ArgumentTypeError(f"Failed to evaluate argument expression '{expr}': {exc}") from exc


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a Helion kernel to JSON.")
    parser.add_argument(
        "--entry",
        required=True,
        help="Kernel entry in the form 'module:function'.",
    )
    parser.add_argument(
        "--arg",
        action="append",
        default=[],
        help="Python expression evaluated in the module scope to provide an example argument.",
    )
    parser.add_argument(
        "--module-name",
        default="helion_module",
        help="Name for the emitted module in the JSON document.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Optional output path. Prints to stdout if omitted.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="Indent level used when pretty-printing JSON.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    ns = parse_args(argv)
    module_name, func_name = _parse_entry(ns.entry)
    module = importlib.import_module(module_name)
    kernel = getattr(module, func_name)
    if not hasattr(kernel, "bind"):
        raise SystemExit(f"{ns.entry} is not a Helion kernel")

    scope = vars(module).copy()
    scope.setdefault("torch", torch)

    example_args = [_eval_arg(expr, scope) for expr in ns.arg]
    spec = export_kernel(kernel, tuple(example_args), module_name=ns.module_name)

    output = json.dumps(spec, indent=ns.indent, sort_keys=True)
    if ns.out:
        ns.out.write_text(output + "\n", encoding="utf-8")
    else:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
