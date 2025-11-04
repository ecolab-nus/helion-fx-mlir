# helion2json

`helion2json` captures Helion kernels after device lowering and re-expresses them in a compact JSON format that preserves tiling structure, tensor slices, and torch math operations. The JSON is the contract between the Python capture stage and the C++ MLIR lowering pipeline.

## Installation

`helion2json` lives inside this repository; installing the project requirements is enough:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r ../requirements.txt
```

The exporter depends on Helion, PyTorch, Triton (on Linux), and `pydantic` for schema validation.

## CLI Usage

```
python -m helion2json --entry module:function [options]
```

Common flags:

| Flag | Description |
| ---- | ----------- |
| `--entry` | Required `module:function` path to the Helion kernel. |
| `--arg` | Python expression evaluated in the module scope to supply example arguments. Repeat for multiple args. |
| `--module-name` | Optional override for the JSON module name (`helion_module` by default). |
| `--out` | If supplied, write JSON to the path instead of stdout. |
| `--indent` | Pretty-print indentation (default 2). |

Example:

```bash
python -m helion2json --entry examples.helion_matmul:matmul \
    --arg "torch.randn(4, 8)" \
    --arg "torch.randn(8, 4)" \
    --out matmul.json
```

## Design Overview

1. **Binding & Block Extraction**  
   - The exporter binds the kernel with example arguments to obtain a `BoundKernel`.  
   - Tile blocks (`block_sizes`) provide names, sizes, and symbolic identifiers (`BlockMeta`).

2. **Graph Translation**  
   - The root FX graph supplies allocation, outer loop launch, and store operations.  
   - Nested `ForLoopGraphInfo` graphs describe tile bodies.  
   - `_ValueEncoder` reconstructs SSA names, converts Helion `load` intrinsics into slice descriptors, and turns torch FX nodes back into `torch.*` JSON ops.

3. **Schema Validation**  
   - The generated dictionary is validated via `helion2json.core.schema.HelionJsonSpec`.  
   - Validation ensures shape expressions, tile metadata, and torch ops conform to the interchange format.

4. **Output**  
   - The final JSON includes `hl.alloc`, `hl.tile.begin/end`, `hl.zeros`, `hl.store_slice`, loop-carried values, and torch op attributes pulled from ATen overloads.

## Programmatic Interface

```python
from helion2json import export_kernel, dump_kernel_json, validate_spec

spec = export_kernel(helion_kernel, (arg0, arg1))
validate_spec(spec)               # optional: already called internally
dump_kernel_json(helion_kernel, "kernel.json", example_args=(arg0, arg1))
```

`export_kernel` will raise descriptive errors if:

- No tensor arguments can be inferred.
- The device IR lacks the expected tiled loop or store pattern.
- Schema validation fails.

## Extending the Exporter

The capture pipeline is intentionally modular:

- Add new helpers in `schema.py` to encode additional intrinsics.  
- Extend `LoopBodyTranslator` or `RootGraphTranslator` with new cases when Helion introduces fresh ops.  
- Update the schema and tests together; `pytest helion2json/tests/test_matmul.py` exercises both success and failure paths.

For cross-referencing, consult `project.plan.md` for the planned coverage matrix and schema examples.
