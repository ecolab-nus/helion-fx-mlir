# Helion FX MLIR Prototype

This repository captures an experimental two-stage lowering pipeline for Helion kernels:

1. **`helion2json`** translates Helion device IR (Python FX graphs) into a normalized JSON interchange format.
2. **`json2mlir`** consumes the JSON payload and produces textual MLIR that preserves PyTorch math as torch dialect ops while expanding Helion tiling constructs into structured loops.

Both stages are designed to be scriptable so new kernels can be inspected quickly without rebuilding large toolchains.

## Repository Layout

```
helion-fx-mlir/
├─ helion2json/          # Python device IR capture → JSON exporter
├─ json2mlir/            # C++ JSON → MLIR translator and CLI
├─ examples/             # Sample Helion kernels used by the tests
├─ samples/              # Shared JSON fixtures (add your own)
└─ project.plan.md       # Detailed design notes and roadmap
```

## Quick Start

### 1. Set up Python environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The requirements file installs PyTorch, Helion, Triton (Linux only), and `pydantic` which `helion2json` uses for schema validation.

### 2. Export Helion kernels to JSON

```bash
python -m helion2json --entry examples.helion_matmul:matmul \
    --arg "torch.randn(4, 8)" --arg "torch.randn(8, 4)" \
    --out samples/matmul.json
```

Refer to `helion2json/README.md` for CLI flags and design notes.

### 3. Build the MLIR translator

```bash
cmake -S json2mlir -B build -G Ninja \
      -DLLVM_DIR=/path/to/llvm/lib/cmake/llvm \
      -DMLIR_DIR=/path/to/llvm/lib/cmake/mlir
ninja -C build json2mlir
```

If you plan to materialize torch dialect ops rather than stubs, add Torch-MLIR to the LLVM toolchain and extend the registry.

### 4. Lower JSON to MLIR

```bash
build/bin/json2mlir samples/matmul.json
```

The command prints MLIR to stdout or you can use `-o output.mlir`.

## Testing

- Python unit tests: `pytest helion2json/tests`
- C++ FileCheck tests: `ninja -C build check-json2mlir` (once the CMake target is wired up)

## Contributing

See `project.plan.md` for the long-term roadmap. The high-priority items are:

- Generalize JSON capture for additional Helion ops beyond matmul.
- Replace the torch call stubs with real torch dialect ops in the MLIR generator.
- Expand the test matrix with dynamic shapes, edge tiles, and error diagnostics.
