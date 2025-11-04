from __future__ import annotations

"""Pydantic models capturing the Helion JSON interchange schema."""

from typing import Any, Dict, List, Mapping, Optional, Union

from pydantic import BaseModel, Field, ValidationError, root_validator, validator

ScalarExpr = Union[int, str]
AllowedDtypes = {"f16", "bf16", "f32", "f64", "i32", "i64"}


class TensorTypeSpec(BaseModel):
    """Tensor shape/element specification used by both tensor and memref types."""

    shape: List[ScalarExpr]
    elem: str

    @validator("elem")
    def validate_elem(cls, value: str) -> str:
        if value not in AllowedDtypes:
            raise ValueError(f"Unsupported element type '{value}'")
        return value


class TypeSpec(BaseModel):
    """Union wrapper that ensures only one of tensor/memref is populated."""

    tensor: Optional[TensorTypeSpec] = None
    memref: Optional[TensorTypeSpec] = None

    @root_validator
    def ensure_single_variant(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        active = [key for key in ("tensor", "memref") if values.get(key) is not None]
        if len(active) != 1:
            raise ValueError("Type spec must define exactly one of {'tensor', 'memref'}")
        return values


class ArgumentSpec(BaseModel):
    """Function argument definition."""

    id: str
    type: TypeSpec


class ReturnSpec(BaseModel):
    """Function return definition."""

    type: TypeSpec


class SliceSpec(BaseModel):
    """Description of a tile slice used by load/store operations."""

    base: str
    offsets: List[ScalarExpr]
    sizes: Optional[List[ScalarExpr]] = None

    @root_validator
    def check_sizes(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        offsets: List[ScalarExpr] = values.get("offsets") or []
        sizes: Optional[List[ScalarExpr]] = values.get("sizes")
        if sizes is not None and len(offsets) != len(sizes):
            raise ValueError("Slice 'offsets' and 'sizes' must have the same length")
        return values


class SliceArg(BaseModel):
    """Helper wrapper so slices can appear inline within torch op argument lists."""

    slice: SliceSpec


class HlAllocOp(BaseModel):
    """Allocate a result buffer (matches Helion's hl.alloc intrinsic)."""

    op: str
    shape: List[ScalarExpr]
    dtype: str
    result: str

    @validator("op")
    def validate_op(cls, value: str) -> str:
        if value != "hl.alloc":
            raise ValueError("hl.alloc op expected")
        return value

    @validator("dtype")
    def validate_dtype(cls, value: str) -> str:
        if value not in AllowedDtypes:
            raise ValueError(f"Unsupported dtype '{value}'")
        return value


class HlTileBeginOp(BaseModel):
    """Start of a tiled loop nest (hl.tile.begin)."""

    op: str
    iters: List[str]
    sizes: List[ScalarExpr]
    result: List[str]

    @validator("op")
    def validate_op(cls, value: str) -> str:
        if value != "hl.tile.begin":
            raise ValueError("hl.tile.begin op expected")
        return value

    @root_validator
    def check_lengths(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        iters = values.get("iters") or []
        sizes = values.get("sizes") or []
        results = values.get("result") or []
        if not (len(iters) == len(sizes) == len(results)):
            raise ValueError("hl.tile.begin requires matching iters/sizes/result lengths")
        return values


class HlTileEndOp(BaseModel):
    """End of the current tiled loop (hl.tile.end)."""

    op: str

    @validator("op")
    def validate_op(cls, value: str) -> str:
        if value != "hl.tile.end":
            raise ValueError("hl.tile.end op expected")
        return value


class HlZerosOp(BaseModel):
    """Zero-initialise a tile-sized scratch buffer (hl.zeros)."""

    op: str
    shape: List[ScalarExpr]
    dtype: str
    result: str

    @validator("op")
    def validate_op(cls, value: str) -> str:
        if value != "hl.zeros":
            raise ValueError("hl.zeros op expected")
        return value

    @validator("dtype")
    def validate_dtype(cls, value: str) -> str:
        if value not in AllowedDtypes:
            raise ValueError(f"Unsupported dtype '{value}'")
        return value


class HlStoreSliceOp(BaseModel):
    """Copy a tile-sized value into the destination buffer (hl.store_slice)."""

    op: str
    dst: str
    offsets: List[ScalarExpr]
    src: str
    sizes: Optional[List[ScalarExpr]] = None

    @validator("op")
    def validate_op(cls, value: str) -> str:
        if value != "hl.store_slice":
            raise ValueError("hl.store_slice op expected")
        return value

    @root_validator
    def check_sizes(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        offsets: List[ScalarExpr] = values.get("offsets") or []
        sizes: Optional[List[ScalarExpr]] = values.get("sizes")
        if sizes is not None and len(offsets) != len(sizes):
            raise ValueError("Offsets and sizes must have the same length")
        return values


class HlReturnOp(BaseModel):
    """Return op at the end of the function body."""

    op: str
    values: List[str]

    @validator("op")
    def validate_op(cls, value: str) -> str:
        if value != "hl.return":
            raise ValueError("hl.return op expected")
        return value


class HlAssertEqOp(BaseModel):
    """Runtime equality assertion (hl.assert_eq)."""

    op: str
    lhs: str
    rhs: str
    msg: str

    @validator("op")
    def validate_op(cls, value: str) -> str:
        if value != "hl.assert_eq":
            raise ValueError("hl.assert_eq op expected")
        return value


class TorchOp(BaseModel):
    """Generic torch.* call emitted directly from captured FX graphs."""

    op: str
    result: Optional[str] = None
    args: List[Union[str, SliceArg, int, float]]
    attrs: Dict[str, Any] = Field(default_factory=dict)

    @validator("op")
    def validate_op(cls, value: str) -> str:
        if not value.startswith("torch."):
            raise ValueError("Torch operations must be prefixed with 'torch.'")
        return value

    @validator("args")
    def validate_args(cls, value: List[Union[str, SliceArg, int, float]]) -> List[Union[str, SliceArg, int, float]]:
        if not value:
            raise ValueError("torch.* ops require at least one argument")
        return value


Operation = Union[
    HlAllocOp,
    HlTileBeginOp,
    HlTileEndOp,
    HlZerosOp,
    HlStoreSliceOp,
    HlReturnOp,
    HlAssertEqOp,
    TorchOp,
]


class FunctionSpec(BaseModel):
    """Top-level function container (name, arguments, body)."""

    name: str
    args: List[ArgumentSpec]
    rets: List[ReturnSpec]
    body: List[Operation]


class ModuleSpec(BaseModel):
    """Module container holding one or more functions."""

    name: str
    funcs: List[FunctionSpec]


class HelionJsonSpec(BaseModel):
    """Root payload wrapper containing the schema version and module."""

    version: int
    module: ModuleSpec


def validate_spec(payload: Mapping[str, Any]) -> HelionJsonSpec:
    """Validate a Helion JSON payload against the interchange schema."""
    return HelionJsonSpec.parse_obj(payload)


__all__ = ["HelionJsonSpec", "validate_spec", "ValidationError"]
