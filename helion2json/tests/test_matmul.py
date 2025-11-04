from __future__ import annotations

import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_ROOT = REPO_ROOT / "examples"
for path in (EXAMPLES_ROOT, PACKAGE_ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import json
import pytest
import torch

import helion_matmul
from helion2json import ValidationError, dump_kernel_json, export_kernel, validate_spec


def test_matmul_kernel_to_json():
    x = torch.randn(4, 8)
    y = torch.randn(8, 4)

    spec = export_kernel(helion_matmul.matmul, (x, y))

    expected = {
        "version": 1,
        "module": {
            "name": "helion_module",
            "funcs": [
                {
                    "name": "matmul",
                    "args": [
                        {"id": "x", "type": {"tensor": {"shape": ["?", "?"], "elem": "f32"}}},
                        {"id": "y", "type": {"tensor": {"shape": ["?", "?"], "elem": "f32"}}},
                    ],
                    "rets": [
                        {"type": {"tensor": {"shape": ["?", "?"], "elem": "f32"}}},
                    ],
                    "body": [
                        {
                            "op": "hl.alloc",
                            "shape": ["dim(x,0)", "dim(y,1)"],
                            "dtype": "f32",
                            "result": "out",
                        },
                        {
                            "op": "hl.tile.begin",
                            "iters": ["m", "n"],
                            "sizes": [4, 4],
                            "result": ["tile_m", "tile_n"],
                        },
                        {
                            "op": "hl.zeros",
                            "shape": ["size(tile_m)", "size(tile_n)"],
                            "dtype": "f32",
                            "result": "acc",
                        },
                        {
                            "op": "hl.tile.begin",
                            "iters": ["k"],
                            "sizes": [8],
                            "result": ["tile_k"],
                        },
                        {
                            "op": "torch.addmm",
                            "result": "acc",
                            "args": [
                                "acc",
                                {
                                    "slice": {
                                        "base": "x",
                                        "offsets": ["tile_m", "tile_k"],
                                        "sizes": ["size(tile_m)", "size(tile_k)"],
                                    }
                                },
                                {
                                    "slice": {
                                        "base": "y",
                                        "offsets": ["tile_k", "tile_n"],
                                        "sizes": ["size(tile_k)", "size(tile_n)"],
                                    }
                                },
                            ],
                            "attrs": {"alpha": 1.0, "beta": 1.0},
                        },
                        {"op": "hl.tile.end"},
                        {
                            "op": "hl.store_slice",
                            "dst": "out",
                            "offsets": ["tile_m", "tile_n"],
                            "src": "acc",
                        },
                        {"op": "hl.tile.end"},
                        {"op": "hl.return", "values": ["out"]},
                    ],
                }
            ],
        },
    }

    assert spec == expected
    # Ensure schema validation passes on the exporter output.
    validate_spec(spec)


def test_dump_kernel_json_reuses_cached_binding(tmp_path: Path):
    x = torch.randn(4, 8)
    y = torch.randn(8, 4)
    kernel = helion_matmul.matmul

    expected = export_kernel(kernel, (x, y))

    output_path = tmp_path / "matmul.json"
    dump_kernel_json(kernel, output_path)

    on_disk = json.loads(output_path.read_text(encoding="utf-8"))
    assert on_disk == expected


def test_validate_spec_rejects_invalid_payload():
    invalid = {"version": 1}
    with pytest.raises(ValidationError):
        validate_spec(invalid)
